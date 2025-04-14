#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : Deep-Crossing.py
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score,recall_score,accuracy_score
import torch.optim as optim

local_path = "../data"

local_click_file = "search_click.csv"
local_click_path_file = os.path.join(local_path, local_click_file)

local_item_file = "item_desc.csv"
local_item_path_file = os.path.join(local_path, local_item_file)

def produce_train_data(input_rating_path_file, input_item_path_file):
    """
    Args:
        input_rating_path_file: user behavior CSV file with columns: userid, item_id, rating, timestamp
        input_item_path_file: path to write the output item vectors in Word2Vec text format
    """
    data = pd.read_csv(input_rating_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "rating"],
                     dtype={"user_id": str, "item_id": str, "rating": float},
                     )
    user_ids = data["user_id"].unique()
    user_2_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

    item_df = pd.read_csv(input_item_path_file,
                     skiprows=1,
                     header=None,
                     names=["item_id", "title"],
                     dtype={"item_id": str, "title": str},
                     )
    item_ids = item_df["item_id"].unique()
    item_2_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    data['user_idx'] = data['user_id'].map(user_2_idx)
    data['item_idx'] = data['item_id'].map(item_2_idx)
    data['rating'] = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())

    global_avg = data["rating"].mean()

    # create training and test datasets
    np.random.seed(0)
    shuffled_index = np.random.permutation(len(data))
    train_size = int(len(data) * 0.9)
    train_index = shuffled_index[:train_size]
    test_index = shuffled_index[train_size:]
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    print(len(train_data), len(test_data))
    return train_data, test_data, user_2_idx, item_2_idx, global_avg

# Define custom Dataset
class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# Define the model
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        x = torch.cat([user_vecs, item_vecs], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for user_batch, item_batch, label_batch in data_loader:
            preds = model(user_batch, item_batch)
            all_preds.extend(preds.numpy())
            all_labels.extend(label_batch.numpy())
    preds_binary = [1 if p >= global_avg else 0 for p in all_preds]
    all_labels = [int(label) for label in all_labels]
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    accuracy = accuracy_score(all_labels, preds_binary)
    return precision, recall, accuracy


# Initialize model, loss function, and optimizer
train_data, test_data, user_2_idx, item_2_idx, global_avg = produce_train_data(local_click_path_file, local_item_path_file)
num_users = len(user_2_idx)
num_items = len(item_2_idx)
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
    train_precision, train_recall, train_accuracy = evaluate(model, train_loader)
    print(f"Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_accuracy:.4f}")

    # Evaluate on test data
    test_precision, test_recall, test_accuracy = evaluate(model, test_loader)
    print(f"Test - Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, Accuracy: {test_accuracy:.4f}")
    print('-' * 80)
