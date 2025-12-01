#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : DeepWide.py (fixed, optimized indexing)
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch.optim as optim
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "view_click.csv"
local_click_path_file = local_path / local_click_file

local_item_file = "item_desc.csv"
local_item_path_file = local_path / local_item_file

local_user_file = "user_desc.csv"
local_user_path_file = local_path / local_user_file 


def load_item_data_file(input_path_file):
    df = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["item_id", "title"],
        dtype={"item_id": str, "title": str},
    )
    df = df.dropna(subset=['item_id'])
    print(df.head(10))
    return df


def load_rating_data_file(input_path_file):
    df = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["user_id", "item_id", "rating"],
        dtype={"user_id": str, "item_id": str, "rating": float},
    )
    df = df.dropna(subset=['user_id', 'item_id'])
    print(df.head(10))
    return df


def load_user_data_file(input_path_file):
    df = pd.read_csv(
        input_path_file,
        skiprows=1,
        header=None,
        names=["user_id", "last_user_agent"],
        dtype={"user_id": str, "last_user_agent": str},
    )
    df = df.dropna(subset=['user_id'])
    print(df.head(10))
    return df


def produce_train_data(input_rating_path_file, input_item_path_file, input_user_path_file):
    raw_user_df = load_user_data_file(input_user_path_file)
    user_id_idx = {v: i for i, v in enumerate(raw_user_df['user_id'].unique())}
    user_last_login_idx = {v: i for i, v in enumerate(raw_user_df['last_user_agent'].unique())}
    user_df = pd.DataFrame({
        'user_id': raw_user_df['user_id'].map(user_id_idx),
        'last_user_agent': raw_user_df['last_user_agent'].map(user_last_login_idx)
    })

    raw_item_df = load_item_data_file(input_item_path_file)
    item_id_idx = {v: i for i, v in enumerate(raw_item_df['item_id'].unique())}
    item_title_idx = {v: i for i, v in enumerate(raw_item_df['title'].unique())}
    item_df = pd.DataFrame({
        'item_id': raw_item_df['item_id'].map(item_id_idx),
        'title': raw_item_df['title'].map(item_title_idx)
    })

    raw_data = load_rating_data_file(input_rating_path_file)
    data = pd.DataFrame({
        'user_id': raw_data['user_id'].map(user_id_idx).fillna(-1).astype(int),
        'item_id': raw_data['item_id'].map(item_id_idx).fillna(-1).astype(int),
        'rating': raw_data['rating']
    })
    data = data[(data['user_id'] != -1) & (data['item_id'] != -1)]

    np.random.seed(0)
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    split = int(len(arr) * 0.9)
    train_set = [tuple(x) for x in data.iloc[arr[:split]].itertuples(index=False, name=None)]
    test_set  = [tuple(x) for x in data.iloc[arr[split:]].itertuples(index=False, name=None)]

    return train_set, test_set, user_df, item_df


def evaluation(y_pred, y_true):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


class Deep_wide(nn.Module):
    def __init__(self, feature_num, user_df, item_df, hidden_dim=64):
        super(Deep_wide, self).__init__()
        self.features = nn.Embedding(feature_num, hidden_dim, max_norm=1)
        # cache DataFrame lookups as tensors for fast indexing
        self.user_features = torch.LongTensor(user_df.values)
        self.item_features = torch.LongTensor(item_df.values)
        total_feature_num = user_df.shape[1] + item_df.shape[1]
        self.mlp_layer = self.__mlp(hidden_dim * total_feature_num)
        self.linear_layer = nn.Linear(hidden_dim * total_feature_num, 1)

    def __mlp(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid())

    def Linear_part(self, feature_embs):
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        return torch.squeeze(self.linear_layer(feature_embs))

    def Deep_part(self, feature_embs):
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        return torch.squeeze(self.mlp_layer(feature_embs))

    def concat_user_item_vec(self, u, i):
        # u, i: LongTensor of indices
        users = self.user_features[u]     # [batch, user_feat]
        items = self.item_features[i]     # [batch, item_feat]
        return torch.cat([users, items], dim=1)

    def forward(self, u, i):
        concat_idx  = self.concat_user_item_vec(u, i)
        concat_embs = self.features(concat_idx)
        lin_out     = self.Linear_part(concat_embs)
        deep_out    = self.Deep_part(concat_embs)
        return torch.sigmoid(lin_out + deep_out)


def train_and_evaluate(local_click_path_file, local_item_path_file, local_user_path_file):
    train_set, test_set, user_df, item_df = produce_train_data(
        local_click_path_file,
        local_item_path_file,
        local_user_path_file
    )

    # Ensure embedding table covers both user and item indices
    max_user_idx = int(user_df.values.max())
    max_item_idx = int(item_df.values.max())
    feature_count = max(max_user_idx, max_item_idx) + 1

    model = Deep_wide(feature_count, user_df, item_df)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    num_epochs = 20

    for epoch in range(num_epochs):
        for user, item, rating in DataLoader(train_set, batch_size=512, shuffle=True):
            optimizer.zero_grad()
            preds = model(user, item)
            loss = loss_fn(preds, rating.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        y_pred = (preds.detach().cpu().numpy() >= 0.5).astype(int)
        y_true = rating.detach().cpu().numpy().astype(int)
        p, r, acc = evaluation(y_pred, y_true)
        print(f"train: Precision {p:.4f} | Recall {r:.4f} | accuracy {acc:.4f}")

        test_arr = np.array(test_set)
        user_test = torch.tensor(test_arr[:, 0], dtype=torch.long)
        item_test = torch.tensor(test_arr[:, 1], dtype=torch.long)
        preds = model(user_test, item_test)
        y_pred = (preds.detach().cpu().numpy() >= 0.5).astype(int)
        y_true = test_arr[:, 2].astype(int)
        p, r, acc = evaluation(y_pred, y_true)
        print(f"test:  Precision {p:.4f} | Recall {r:.4f} | accuracy {acc:.4f}")
        print('-' * 88)


if __name__ == "__main__":
    train_and_evaluate(local_click_path_file, local_item_path_file, local_user_path_file)
