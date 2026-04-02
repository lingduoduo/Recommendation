import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# File Paths
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "model", "train_gift_two_tower_pt")


def resolve_data_file(filename):
    candidate_paths = [
        os.path.join(REPO_ROOT, "data", "input", filename),
        os.path.join(SCRIPT_DIR, "..", "data", "input", filename),
        os.path.join(os.getcwd(), "data", "input", filename),
        os.path.join(os.getcwd(), filename),
    ]

    for path in candidate_paths:
        normalized_path = os.path.abspath(path)
        if os.path.exists(normalized_path):
            return normalized_path

    formatted_candidates = "\n".join(
        f"  - {os.path.abspath(path)}" for path in candidate_paths
    )
    raise FileNotFoundError(
        f"Could not find required data file '{filename}'. Checked:\n{formatted_candidates}"
    )

def get_default_data_paths():
    return {
        "click_path": resolve_data_file("search_click.csv"),
        "item_path": resolve_data_file("item_desc.csv"),
    }

# ----------------------------
# Data Loading
# ----------------------------
def load_item_data_file(path):
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=["item_id", "title"],
                     dtype={"item_id": str, "title": str})
    print(df.head(10))
    return df

def load_click_data_file(path):
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=["user_id", "item_id", "rating"],
                     dtype={"user_id": str, "item_id": str, "rating": float})
    print(df.head(10))
    return df

# ----------------------------
# Dataset
# ----------------------------
class RatingDataset(Dataset):
    def __init__(self, df):
        self.user_id = df["user_id"].values
        self.item_id = df["item_id"].values

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_id[idx],
            "item_id": self.item_id[idx],
        }

# ----------------------------
# Embedding Layers
# ----------------------------
class ItemModel(nn.Module):
    def __init__(self, unique_item_ids, embedding_dim=96):
        super().__init__()
        self.lookup = {v: i + 1 for i, v in enumerate(unique_item_ids)}
        self.embedding = nn.Embedding(len(self.lookup) + 1, embedding_dim)

    def forward(self, item_ids):
        if isinstance(item_ids, torch.Tensor):
            return self.embedding(item_ids)
        indices = [self.lookup.get(v, 0) for v in item_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(idx_tensor)

class UserModel(nn.Module):
    def __init__(self, unique_user_ids, embedding_dim=96):
        super().__init__()
        self.lookup = {v: i + 1 for i, v in enumerate(unique_user_ids)}
        self.embedding = nn.Embedding(len(self.lookup) + 1, embedding_dim)

    def forward(self, user_ids):
        if isinstance(user_ids, torch.Tensor):
            return self.embedding(user_ids)
        indices = [self.lookup.get(v, 0) for v in user_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(idx_tensor)

# ----------------------------
# DSSM Components
# ----------------------------
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        layers = []
        for i in range(len(hidden_units)):
            in_dim = input_dim if i == 0 else hidden_units[i - 1]
            out_dim = hidden_units[i]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CosineSimilarity(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb, item_emb):
        sim = torch.nn.functional.cosine_similarity(user_emb, item_emb, dim=-1)
        return sim / self.temperature

class PredictLayer(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).unsqueeze(-1)

# ----------------------------
# DSSM Model
# ----------------------------
class DSSM(nn.Module):
    def __init__(self, user_model, item_model, embedding_dim=96, dnn_units=[64, 32], temperature=10.0):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.user_dnn = DNN(embedding_dim, dnn_units)
        self.item_dnn = DNN(embedding_dim, dnn_units)
        self.similarity = CosineSimilarity(temperature)
        self.predict = PredictLayer()

    def forward(self, batch):
        user_raw = self.user_model(batch["user_id"])
        item_raw = self.item_model(batch["item_id"])
        user_embed = self.user_dnn(user_raw)
        item_embed = self.item_dnn(item_raw)
        sim = self.similarity(user_embed, item_embed)
        return self.predict(sim)  # shape [B, 1]

    def training_step(self, batch, optimizer, loss_fn):
        self.train()
        optimizer.zero_grad()
        output = self.forward(batch)
        labels = torch.ones_like(output)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        return loss.item()


def build_item_title_lookup(item_df):
    return {
        str(row["item_id"]): str(row["title"])
        for _, row in item_df.iterrows()
    }


def build_item_popularity(click_df):
    popularity = click_df["item_id"].value_counts()
    return {str(item_id): float(score) for item_id, score in popularity.items()}


def save_model_bundle(
    model,
    user_ids,
    item_ids,
    item_title_lookup,
    item_popularity,
    model_dir=DEFAULT_MODEL_DIR,
    embedding_dim=96,
    dnn_units=None,
    temperature=10.0,
):
    if dnn_units is None:
        dnn_units = [64, 32]

    os.makedirs(model_dir, exist_ok=True)
    bundle_path = os.path.join(model_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "user_ids": list(user_ids),
            "item_ids": list(item_ids),
            "item_title_lookup": dict(item_title_lookup),
            "item_popularity": dict(item_popularity),
            "embedding_dim": embedding_dim,
            "dnn_units": list(dnn_units),
            "temperature": temperature,
        },
        bundle_path,
    )
    return bundle_path


def load_model_bundle(model_dir=DEFAULT_MODEL_DIR):
    bundle_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Could not find trained model artifact at {bundle_path}"
        )

    bundle = torch.load(bundle_path, map_location="cpu")
    user_model = UserModel(bundle["user_ids"], embedding_dim=bundle["embedding_dim"])
    item_model = ItemModel(bundle["item_ids"], embedding_dim=bundle["embedding_dim"])
    model = DSSM(
        user_model,
        item_model,
        embedding_dim=bundle["embedding_dim"],
        dnn_units=bundle["dnn_units"],
        temperature=bundle["temperature"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return bundle, model


def score_items_for_user(model, user_id, item_ids):
    with torch.no_grad():
        batch = {
            "user_id": [user_id] * len(item_ids),
            "item_id": list(item_ids),
        }
        scores = model(batch).squeeze(-1).cpu().numpy()
    return scores


def recommend_for_user(model, bundle, user_id, top_k=5):
    item_ids = bundle["item_ids"]
    item_title_lookup = bundle.get("item_title_lookup", {})
    known_user_ids = set(bundle["user_ids"])

    if user_id in known_user_ids:
        scores = score_items_for_user(model, user_id, item_ids)
    else:
        popularity = bundle.get("item_popularity", {})
        scores = np.array([popularity.get(item_id, 0.0) for item_id in item_ids], dtype=float)

    ranked_indices = np.argsort(-scores)[:top_k]
    recommendations = []
    for idx in ranked_indices:
        item_id = item_ids[idx]
        recommendations.append(
            {
                "item_id": item_id,
                "title": item_title_lookup.get(item_id, item_id),
                "score": float(scores[idx]),
            }
        )
    return recommendations

# ----------------------------
# Training & Evaluation
# ----------------------------
def train_and_evaluate(
    click_path=None,
    item_path=None,
    save_model=True,
    model_dir=DEFAULT_MODEL_DIR,
):
    data_paths = get_default_data_paths()
    click_path = click_path or data_paths["click_path"]
    item_path = item_path or data_paths["item_path"]

    # Load data
    item_df = load_item_data_file(item_path)
    click_df = load_click_data_file(click_path)

    embedding_dim = 96
    dnn_units = [64, 32]
    temperature = 10.0

    # Train/test split
    train_df, test_df = train_test_split(click_df, test_size=0.2, random_state=42)
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    # Models
    user_model = UserModel(click_df["user_id"].unique())
    item_model = ItemModel(item_df["item_id"].astype(str).unique(), embedding_dim=embedding_dim)
    model = DSSM(
        user_model,
        item_model,
        embedding_dim=embedding_dim,
        dnn_units=dnn_units,
        temperature=temperature,
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            loss = model.training_step(batch, optimizer, loss_fn)
            total_loss += loss
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch).squeeze().numpy()
            all_preds.extend(preds)
            all_labels.extend([1.0] * len(preds))  # All are implicit positives

    thresholded = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, thresholded)
    print(f"Test Accuracy: {accuracy:.4f}")
    bundle_path = None
    if save_model:
        bundle_path = save_model_bundle(
            model=model,
            user_ids=click_df["user_id"].astype(str).unique().tolist(),
            item_ids=item_df["item_id"].astype(str).unique().tolist(),
            item_title_lookup=build_item_title_lookup(item_df),
            item_popularity=build_item_popularity(click_df),
            model_dir=model_dir,
            embedding_dim=embedding_dim,
            dnn_units=dnn_units,
            temperature=temperature,
        )
        print(f"Saved model bundle to {bundle_path}")

    return {
        "model": model,
        "accuracy": accuracy,
        "bundle_path": bundle_path,
        "model_dir": model_dir,
    }

# Run training and evaluation
if __name__ == "__main__":
    train_and_evaluate()
