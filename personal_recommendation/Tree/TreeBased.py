#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : TreeBased.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "view_click.csv"
local_click_path_file = local_path / local_click_file

local_item_file = "item_desc.csv"
local_item_path_file = local_path / local_item_file

score_thr = 1


def produce_train_data(input_rating_path_file, input_item_path_file):
    """
    Args:
        input_rating_path_file: user behavior CSV file with columns: userid, item_id, rating, timestamp
        input_item_path_file: item CSV file with columns: item_id, title
    Returns:
        df: DataFrame with columns: user_id, item_id, rating, title
    """
    rating_df = pd.read_csv(input_rating_path_file,
                            skiprows=1,
                            header=None,
                            names=["user_id", "item_id", "rating"],
                            dtype={"user_id": str, "item_id": str, "rating": float},
                            )

    item_df = pd.read_csv(input_item_path_file,
                          skiprows=1,
                          header=None,
                          names=["item_id", "title"],
                          dtype={"item_id": str, "title": str},
                          )

    df = rating_df.merge(item_df, how="inner", on="item_id")
    print(len(df))
    return df


# Data Transformation
def transformation(df: pd.DataFrame):
    """
    Args:
        df: DataFrame to be transformed
    Returns:
        cat_transform: categorical transformation pipeline
        num_transform: numerical transformation pipeline
    """
    # Identify categorical columns
    c_cols = df.select_dtypes(include=['object', 'category']).columns
    print(c_cols)

    # Categorical transformation pipeline
    c_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    cat_transform = [("onehot", c_pipeline, c_cols)]
    print(cat_transform)

    # Identify numerical columns
    n_cols = df.select_dtypes(include=[np.number]).columns.difference(['rating'])
    n_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
        ("imputer", SimpleImputer(strategy='mean')),
        ("standardizer", StandardScaler()),
    ])
    num_transform = [("numerical", n_pipeline, n_cols)]
    print(num_transform)
    return cat_transform, num_transform


def model(data, cat_transform, num_transform):
    """
    Args:
        train: training data
        test: test data
    Returns:
        model: trained model
    """
    target_col = "rating"
    split_X = data.drop([target_col], axis=1)
    split_y = data[target_col] >= score_thr
    train_x, test_x, train_y, test_y = train_test_split(split_X, split_y, train_size=0.9, random_state=149849802,
                                                        stratify=split_y)
    # Define the model pipeline
    preprocessor = ColumnTransformer(num_transform + cat_transform, remainder="passthrough", sparse_threshold=1)

    # Prepare the pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier())
    ])
    pipeline.fit(train_x, train_y)

    # Fit the model
    pred1 = pipeline.predict(train_x)
    accuracy1 = accuracy_score(train_y, pred1)
    print('Accuracy in training data: %.4f' % accuracy1)
    print('Precision in training data:', precision_score(train_y, pred1))
    print('Recall in training data:', recall_score(train_y, pred1))
    print('F1-score in training data:', f1_score(train_y, pred1))
    prob1 = pipeline.predict_proba(train_x)[:, 1]
    print('AUC in training data:', roc_auc_score(train_y, prob1))

    pred2 = pipeline.predict(test_x)
    accuracy2 = accuracy_score(test_y, pred2)
    print('Accuracy in test data: %.4f' % accuracy2)
    print('Precision in test data:', precision_score(test_y, pred2))
    print('Recall in test data:', recall_score(test_y, pred2))
    print('F1-score in test data:', f1_score(test_y, pred2))
    prob2 = pipeline.predict_proba(test_x)[:, 1]
    print('AUC in test data:', roc_auc_score(test_y, prob2))


if __name__ == "__main__":
    data = produce_train_data(local_click_path_file, local_item_path_file)
    c_trans, n_trans = transformation(data)
    model(data, c_trans, n_trans)
