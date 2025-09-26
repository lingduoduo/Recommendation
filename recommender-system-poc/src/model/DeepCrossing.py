#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : DeepCrossing.py
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.optim as optim
from pathlib import Path

import sys
ROOT_DIR = Path.cwd().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils.evaluation import compute_evaluation_metrics

local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "view_click.csv"
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
    df = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["item_id", "title"],
        dtype={"item_id": str, "title": str},
    )
    df = df.dropna(subset=['item_id'])
    # df = df.sample(n=10000)
    print(df.head(10))
    return df


def load_rating_data_file(input_path_file):
    """
    Args:
        input_path_file: user rating file
    Returns:
        data: DataFrame with columns: user_id, item_id, rating
    """
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "rating"],
                     dtype={"user_id": str, "item_id": str, "rating": float},
                     )
    df = df.dropna(subset=['user_id'])
    df = df.dropna(subset=['item_id'])
    # df = df.sample(n=10000)
    print(df.head(10))
    return df


def produce_train_data(input_rating_path_file, input_item_path_file):
    """
    Args:
        input_rating_path_file: user behavior CSV file with columns: userid, item_id, rating, timestamp
        input_item_path_file: path to write the output item vectors in Word2Vec text format
    Returns:
        train_data: DataFrame with columns: user_id, item_id, rating
        test_data: DataFrame with columns: user_id, item_id, rating
        user_2_idx: dict mapping user_id to index
        item_2_idx: dict mapping item_id to index
        global_avg: float, global average rating
    """
    raw_item_df = load_item_data_file(input_item_path_file)
    item_df = pd.DataFrame()
    item_to_idx = {v: i for i, v in enumerate(raw_item_df['item_id'].unique())}
    item_df["item_id"] = raw_item_df["item_id"].map(item_to_idx)
    item_title_idx = {v: i for i, v in enumerate(raw_item_df['title'].unique())}
    item_df["title"] = raw_item_df["title"].map(item_title_idx)

    raw_data = load_rating_data_file(input_rating_path_file)
    user_ids = raw_data["user_id"].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    data = pd.DataFrame()
    data['user_id'] = raw_data['user_id'].map(user_to_idx).fillna(-1).astype(int)
    data['item_id'] = raw_data['item_id'].map(item_to_idx).fillna(-1).astype(int)
    data['rating'] = raw_data['rating']
    data = data[data['user_id'] != -1]
    data = data[data['item_id'] != -1]

    global_avg = data["rating"].mean()

    # create training and test datasets
    np.random.seed(0)
    shuffled_index = np.random.permutation(len(data))
    train_size = int(len(data) * 0.8)
    train_index = shuffled_index[:train_size]
    test_index = shuffled_index[train_size:]
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    print(len(train_data), len(test_data))
    return train_data, test_data, user_to_idx, item_to_idx, global_avg


# Define custom Dataset
class InteractionDataset(Dataset):
    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame containing user_idx, item_idx, and rating columns.
        Returns:
            None
        """
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.labels = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        """
        Args: None
        Returns:
            Length of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the data point.
        Returns:
            Tuple of user index, item index, and rating.
        """
        return self.users[idx], self.items[idx], self.labels[idx]


# Define the model
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        """
        Args:
            num_users: number of users
            num_items: number of items
            embedding_dim: embedding dimension
        Returns:
            None
        """
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, user_indices, item_indices):
        """
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
        Returns:
            Tensor of predicted ratings
        """
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        x = torch.cat([user_vecs, item_vecs], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()


# Evaluation function
def evaluate(model, data_loader, global_avg):
    """
    Args:
        model: trained model
        data_loader: DataLoader for test data
        global_avg: global average rating
    Returns:
        precision: precision score
        recall: recall score
        accuracy: accuracy score
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for user_batch, item_batch, label_batch in data_loader:
            preds = model(user_batch, item_batch)
            all_preds.extend(preds.numpy())
            all_labels.extend(label_batch.numpy())
    y_pred = [1 if p >= global_avg else 0 for p in all_preds]
    y_true = [int(label) for label in all_labels]
    precision, recall, acc = compute_evaluation_metrics(y_true, y_pred)
    return precision, recall, acc


def train_and_evaluate(local_click_path_file, local_item_path_file):
    """
    Args:
        local_click_path_file: path to user behavior CSV file
        local_item_path_file: path to item description CSV file
    Returns:
        None
    """
    # Load data and create datasets
    train_data, test_data, user_to_idx, item_to_idx, global_avg = produce_train_data(local_click_path_file,
                                                                                     local_item_path_file)
    num_users = len(user_to_idx)
    num_items = len(item_to_idx)
    embedding_dim = 32

    model = RecommenderModel(num_users, num_items, embedding_dim)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Prepare DataLoaders
    train_dataset = InteractionDataset(train_data)
    test_dataset = InteractionDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for user_batch, item_batch, label_batch in train_loader:
            optimizer.zero_grad()
            preds = model(user_batch, item_batch)
            loss = loss_fn(preds, label_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Evaluate on training data
        train_precision, train_recall, train_accuracy = evaluate(model, train_loader, global_avg)
        print(f"Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_accuracy:.4f}")

        # Evaluate on test data
        test_precision, test_recall, test_accuracy = evaluate(model, test_loader, global_avg)
        print(f"Test - Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, Accuracy: {test_accuracy:.4f}")
        print('-' * 80)


if __name__ == "__main__":
    train_and_evaluate(local_click_path_file, local_item_path_file)
