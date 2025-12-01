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
local_path = "../data/input"
local_click_path = os.path.join(local_path, "search_click.csv")
local_item_path = os.path.join(local_path, "item_desc.csv")

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
        indices = [self.lookup.get(v, 0) for v in item_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(idx_tensor)

class UserModel(nn.Module):
    def __init__(self, unique_user_ids, embedding_dim=96):
        super().__init__()
        self.lookup = {v: i + 1 for i, v in enumerate(unique_user_ids)}
        self.embedding = nn.Embedding(len(self.lookup) + 1, embedding_dim)

    def forward(self, user_ids):
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

# ----------------------------
# Training & Evaluation
# ----------------------------
def train_and_evaluate():
    # Load data
    item_df = load_item_data_file(local_item_path)
    click_df = load_click_data_file(local_click_path)

    # Train/test split
    train_df, test_df = train_test_split(click_df, test_size=0.2, random_state=42)
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    # Models
    user_model = UserModel(click_df["user_id"].unique())
    item_model = ItemModel(click_df["item_id"].unique())
    model = DSSM(user_model, item_model)

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

# Run training and evaluation
if __name__ == "__main__":
    train_and_evaluate()
