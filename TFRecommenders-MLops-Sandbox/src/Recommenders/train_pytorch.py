#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : DSSM.py
"""
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

local_path = "../data/input"
local_click_file = "search_click.csv"
local_click_path_file = os.path.join(local_path, local_click_file)
local_item_file = "item_desc.csv"
local_item_path_file = os.path.join(local_path, local_item_file)

def load_item_data_file(input_path_file):
    df = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["item_id", "title"],
        dtype={"item_id": str, "title": str},
    )
    print(df.head(10))
    return df

def load_click_data_file(input_path_file):
    df = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["user_id", "item_id", "rating"],
        dtype={"user_id": str, "item_id": str, "rating": float},
    )
    print(df.head(10))
    return df

class RatingDataset(Dataset):
    def __init__(self, df, user_lookup, item_lookup):
        # map strings to integer indices; unseen IDs map to 0
        self.user_idx = torch.tensor(
            [user_lookup.get(u, 0) for u in df["user_id"].values],
            dtype=torch.long
        )
        self.item_idx = torch.tensor(
            [item_lookup.get(i, 0) for i in df["item_id"].values],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_idx[idx],
            "item_id": self.item_idx[idx],
        }

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

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        layers = []
        for i, h in enumerate(hidden_units):
            in_dim = input_dim if i == 0 else hidden_units[i - 1]
            layers.append(nn.Linear(in_dim, h))
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
        return self.predict(sim)

    def training_step(self, batch, optimizer, loss_fn):
        self.train()
        optimizer.zero_grad()
        output = self.forward(batch)
        labels = torch.ones_like(output)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

def train_and_evaluate(local_item_path=local_item_path_file, local_click_path=local_click_path_file):
    # Load data
    item_df = load_item_data_file(local_item_path)
    click_df = load_click_data_file(local_click_path)

    # Build lookups from all known IDs
    unique_users = click_df["user_id"].unique()
    unique_items = item_df["item_id"].unique()
    user_lookup = {v: i+1 for i, v in enumerate(unique_users)}
    item_lookup = {v: i+1 for i, v in enumerate(unique_items)}

    # Split and wrap
    train_df, test_df = train_test_split(click_df, test_size=0.2, random_state=42)
    train_dataset = RatingDataset(train_df, user_lookup, item_lookup)
    test_dataset  = RatingDataset(test_df,  user_lookup, item_lookup)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=512)

    # Instantiate models
    user_model = UserModel(unique_users)
    item_model = ItemModel(unique_items)
    model = DSSM(user_model, item_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            loss = model.training_step(batch, optimizer, loss_fn)
            total_loss += loss
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation (original + retrieval metrics)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch).squeeze().numpy()
            all_preds.extend(preds)
            all_labels.extend([1.0] * len(preds))

    # classification accuracy
    thresholded = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, thresholded)
    print(f"Test Accuracy: {accuracy:.4f}")

    # retrieval metrics
    print("Computing retrieval metrics...")
    # precompute item embeddings
    all_item_idx = torch.arange(1, len(unique_items)+1, dtype=torch.long)
    with torch.no_grad():
        item_raw = item_model(all_item_idx)
        item_embeds = model.item_dnn(item_raw)
    # embed test users
    test_user_idx = test_dataset.user_idx
    test_item_idx = test_dataset.item_idx.tolist()
    with torch.no_grad():
        user_raw = user_model(test_user_idx)
        user_embeds = model.user_dnn(user_raw)
    # pairwise cosine
    sims = torch.nn.functional.cosine_similarity(
        user_embeds.unsqueeze(1), item_embeds.unsqueeze(0), dim=-1
    )  # [n_test, n_items]
    ranks = []
    for i, true_idx in enumerate(test_item_idx):
        row = sims[i]
        rank = (row > row[true_idx-1]).sum().item() + 1
        ranks.append(rank)
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    recall1 = sum(r <= 1 for r in ranks) / len(ranks)
    recall5 = sum(r <= 5 for r in ranks) / len(ranks)
    recall10 = sum(r <= 10 for r in ranks) / len(ranks)
    print(f"MRR: {mrr:.4f}, R@1: {recall1:.4f}, R@5: {recall5:.4f}, R@10: {recall10:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
