#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : Deep-Crossing.py
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch.optim as optim

# Load data
data = pd.read_csv("../data/search_click.csv")
item_df = pd.read_csv("../data/item_desc.csv")

# Map user and item IDs to integer indices
user_ids = data["user_id"].unique()
user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_ids = item_df["item_id"].unique()
item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

# Map IDs in the main data
data["user_idx"] = data["user_id"].map(user2idx)
data["item_idx"] = data["item_id"].map(item2idx)

# Split data into training and testing sets
np.random.seed(0)
shuffled_indices = np.random.permutation(len(data))
train_size = int(0.9 * len(data))
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]

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

# Initialize model, loss function, and optimizer
num_users = len(user2idx)
num_items = len(item2idx)
embedding_dim = 32

model = RecommenderModel(num_users, num_items, embedding_dim)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare DataLoaders
train_dataset = InteractionDataset(train_data)
test_dataset = InteractionDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512)

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
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    precision = precision_score(all_labels, preds_binary)
    recall = recall_score(all_labels, preds_binary)
    accuracy = accuracy_score(all_labels, preds_binary)
    return precision, recall, accuracy

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
