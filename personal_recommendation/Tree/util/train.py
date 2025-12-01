# train_lgbm.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import mlflow
from ray import tune

ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"
local_click_path_file = local_path / "view_click.csv"
local_item_path_file = local_path / "item_desc.csv"
score_thr = 1

def produce_train_data():
    rating_df = pd.read_csv(local_click_path_file, skiprows=1, header=None,
                            names=["user_id", "item_id", "rating"],
                            dtype={"user_id": str, "item_id": str, "rating": float})
    item_df = pd.read_csv(local_item_path_file, skiprows=1, header=None,
                          names=["item_id", "title"],
                          dtype={"item_id": str, "title": str})
    return rating_df.merge(item_df, on="item_id")

def get_transformers(df):
    c_cols = df.select_dtypes(include=['object', 'category']).columns
    c_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("one_hot", OneHotEncoder(handle_unknown='ignore'))
    ])
    n_cols = df.select_dtypes(include=[np.number]).columns.difference(['rating'])
    n_pipeline = Pipeline([
        ("convert", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler())
    ])
    return [("cat", c_pipeline, c_cols)], [("num", n_pipeline, n_cols)]

def train_model_tune(config, data=None, cat_trans=None, num_trans=None):
    mlflow.set_experiment("lgbm_ray_tune")
    with mlflow.start_run(nested=True):
        mlflow.log_params(config)

        X = data.drop(columns=["rating"])
        y = data["rating"] >= score_thr
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        preprocessor = ColumnTransformer(num_trans + cat_trans, remainder="passthrough", sparse_threshold=1)
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("clf", LGBMClassifier(**config))
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob)

        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
        tune.report(mean_accuracy=acc, roc_auc=auc)
