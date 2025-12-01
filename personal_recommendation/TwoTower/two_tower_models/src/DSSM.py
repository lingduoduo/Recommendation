#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : DSSM.py
"""
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "search_click.csv"
local_click_path_file = local_path / local_click_file

local_item_file = "item_desc.csv"
local_item_path_file = local_path / local_item_file


def load_item_data_file(input_path_file):
    """
    Args:
        input_path_file: user rating file
    Returns:
        data: DataFrame with columns: item_id, title
    """
    data = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["item_id", "title"],
        dtype={"item_id": str, "title": str},
    )
    data = data.dropna(subset=['item_id'])
    # data = data.sample(n=10000)
    print(data.head(10))
    return data


def load_click_data_file(input_path_file):
    """
    Args:
        input_path_file: user rating file
    Returns:
        data: DataFrame with columns: user_id, item_id, rating
    """
    data = pd.read_csv(input_path_file,
                       skiprows=1,
                       header=None,
                       names=["user_id", "item_id", "rating"],
                       dtype={"user_id": str, "item_id": str, "rating": float},
                       )
    data = data.dropna(subset=['user_id'])
    data = data.dropna(subset=['item_id'])
    # data = data.sample(n=10000)
    print(data.head(10))
    return data


class RatingDataset(Dataset):
    def __init__(self, df, user_lookup, item_lookup):
        """
        Args:
           df: DataFrame with columns: user_id, item_id
           user_lookup: dict mapping user_id to integer index
           item_lookup: dict mapping item_id to integer index
        Returns:
            None
        """
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
        """
        Args:
            None
        Returns:
            length of the dataset
        """
        return len(self.user_idx)

    def __getitem__(self, idx):
        """
        Args:
            idx: index of the item to retrieve
        Returns:
            dict with user_id and item_id based on dataframe idx
        """
        return {
            "user_id": self.user_idx[idx],
            "item_id": self.item_idx[idx],
        }


class ItemModel(nn.Module):
    def __init__(self, unique_item_ids, embedding_dim=96):
        """
        Args:
            unique_item_ids: list of unique item IDs
            embedding_dim: embedding dimension
        Returns:
            None
        """
        super().__init__()
        self.lookup = {v: i + 1 for i, v in enumerate(unique_item_ids)}
        self.embedding = nn.Embedding(len(self.lookup) + 1, embedding_dim)

    def forward(self, item_ids):
        """
        Args:
            item_ids: list of item IDs
        Returns:
            item embeddings for the given item IDs
        """
        if isinstance(item_ids, torch.Tensor):
            return self.embedding(item_ids)

        indices = [self.lookup.get(v, 0) for v in item_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(idx_tensor)


class UserModel(nn.Module):
    def __init__(self, user_ids, embedding_dim=96):
        """
        Args:
            user_ids: list of unique user IDs
        Returns:
            None
        """
        super().__init__()
        self.lookup = {v: i + 1 for i, v in enumerate(user_ids)}
        self.embedding = nn.Embedding(len(self.lookup) + 1, embedding_dim)

    def forward(self, user_ids):
        """
        Args:
            user_ids: list of user IDs
        Returns:
            user embeddings for the given user IDs
        """
        if isinstance(user_ids, torch.Tensor):
            return self.embedding(user_ids)
        indices = [self.lookup.get(v, 0) for v in user_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(idx_tensor)


class TwoTowers(nn.Module):
    def __init__(self, user_model, item_model):
        """
        Args:
            user_model: user model as input
            item_model: item model as input
        Returns:
            None
        """
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model

    def forward(self, batch):
        """
        Args:
            batch: batch of data
        Returns:
            user embeddings and item embeddings
        """
        user_embeds = self.user_model(batch["user_id"])
        item_embeds = self.item_model(batch["item_id"])
        return user_embeds, item_embeds


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        """
        Args:
            input_dim: input dimension
            hidden_units: list of hidden units
        Returns:
            None
        """
        super().__init__()
        layers = []
        for i in range(len(hidden_units)):
            in_dim = input_dim if i == 0 else hidden_units[i - 1]
            out_dim = hidden_units[i]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: input tensor
        Returns:
            output tensor
        """
        return self.network(x)


class CosineSimilarity(nn.Module):
    def __init__(self, temperature=1.0):
        """
        Args:
            temperature: scaling factor for similarity
        Returns:
            None
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb, item_emb):
        """
        Args:
            user_emb: user embeddings
            item_emb: item embeddings
        Returns:
            similarity score between user and item embeddings
        """
        sim = torch.nn.functional.cosine_similarity(user_emb, item_emb, dim=-1)
        return sim / self.temperature


class PredictLayer(nn.Module):
    def forward(self, x):
        """
        Args:
            x: input tensor
        Returns:
            output tensor after applying sigmoid activation
        """
        return torch.sigmoid(x).unsqueeze(-1)


class DSSM(nn.Module):
    def __init__(self, user_model, item_model, embedding_dim=96, dnn_units=[64, 32], temperature=1.0):
        """
        Args:
            user_model: user model as input
            item_model: item model as input
            embedding_dim: embedding dimension
            dnn_units: list dimension of hidden units
            temperature: scaling factor for similarity
        Returns:
            None
        """
        super().__init__()
        self.base = TwoTowers(user_model, item_model)
        self.user_dnn = DNN(embedding_dim, dnn_units)
        self.item_dnn = DNN(embedding_dim, dnn_units)
        self.similarity = CosineSimilarity(temperature)
        self.predict = PredictLayer()

    def forward(self, batch):
        """
        Args:
            batch: batch of data
        Returns:
            prediction score for the given batch
        """
        user_raw, item_raw = self.base(batch)
        user_embed = self.user_dnn(user_raw)
        item_embed = self.item_dnn(item_raw)
        sim = self.similarity(user_embed, item_embed)
        return self.predict(sim)

    def training_step(self, batch, optimizer, loss_fn):
        """
        Args:
            batch: batch of data
            optimizer: optimizer for the model
            loss_fn: loss function
        Returns:
            loss: computed loss value
        """
        self.train()
        optimizer.zero_grad()
        output = self.forward(batch)
        labels = torch.ones_like(output)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        return loss.item()


def train_and_evaluate(local_item_path=local_item_path_file, local_click_path=local_click_path_file):
    """
    Args:
        local_item_path: path to item data file
        local_click_path: path to click data file
    Returns:
        None
    """
    # 1) load raw CSVs
    item_df = load_item_data_file(local_item_path)
    click_df = load_click_data_file(local_click_path)

    # 2) build consistent lookups from raw IDs → ints (1...N)
    unique_users = click_df["user_id"].unique()
    unique_items = item_df["item_id"].unique()
    user_lookup = {v: i + 1 for i, v in enumerate(unique_users)}
    item_lookup = {v: i + 1 for i, v in enumerate(unique_items)}

    # 3) split & wrap into new Dataset that returns ints, not strs
    train_df, test_df = train_test_split(click_df, test_size=0.2, random_state=42)
    train_dataset = RatingDataset(train_df, user_lookup, item_lookup)
    test_dataset = RatingDataset(test_df, user_lookup, item_lookup)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    # 4) instantiate your models exactly as before
    user_model = UserModel(unique_users)
    item_model = ItemModel(unique_items)
    model = DSSM(user_model, item_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # 5) training
    for epoch in range(20):
        total_loss = 0.0
        for batch in train_loader:
            loss = model.training_step(batch, optimizer, loss_fn)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader):.4f}")

    # 6) evaluation
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
    all_item_idx = torch.arange(1, len(unique_items) + 1, dtype=torch.long)
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
        rank = (row > row[true_idx - 1]).sum().item() + 1
        ranks.append(rank)
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    recall1 = sum(r <= 1 for r in ranks) / len(ranks)
    recall5 = sum(r <= 5 for r in ranks) / len(ranks)
    recall10 = sum(r <= 10 for r in ranks) / len(ranks)
    recall50 = sum(r <= 50 for r in ranks) / len(ranks)
    print(f"MRR: {mrr:.4f}, R@1: {recall1:.4f}, R@5: {recall5:.4f}, R@10: {recall10:.4f}, R@50: {recall50:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
