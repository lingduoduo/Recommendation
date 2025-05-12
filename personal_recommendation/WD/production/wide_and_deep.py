#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : DeepWide.py
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score

from torch.utils.data import DataLoader
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


def load_user_data_file(input_path_file):
    """
    Args:
        input_path_file: user rating file
    Returns:
        df: DataFrame with columns: user_id, last_user_agent
    """
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "last_user_agent"],
                     dtype={"user_id": str, "last_user_agent": str},
                     )
    df = df.dropna(subset=['user_id'])
    # df = df.sample(n=10000)
    print(df.head(10))
    return df


def produce_train_data(input_rating_path_file, input_item_path_file, input_user_path_file):
    """
    Args:
        input_rating_path_file: user behavior CSV file with columns: userid, item_id, rating, timestamp
        input_item_path_file: item CSV file with columns: item_id, title
        input_user_path_file: user CSV file with columns: user_id, last_user_agent
    Returns:
        train_data: DataFrame with columns: user_id, item_id, rating
        test_data: DataFrame with columns: user_id, item_id, rating
        user_df: DataFrame with columns: user_id, last_user_agent
        item_df: DataFrame with columns: item_id, title
    """
    raw_user_df = load_user_data_file(input_user_path_file)
    user_df = pd.DataFrame()
    user_id_idx = {v: i for i, v in enumerate(raw_user_df['user_id'].unique())}
    user_df["user_id"] = raw_user_df["user_id"].map(user_id_idx)
    user_last_login_idx = {v: i for i, v in enumerate(raw_user_df['last_user_agent'].unique())}
    user_df["last_user_agent"] = raw_user_df["last_user_agent"].map(user_last_login_idx)
    print(user_df)

    raw_item_df = load_item_data_file(input_item_path_file)
    item_df = pd.DataFrame()
    item_id_idx = {v: i for i, v in enumerate(raw_item_df['item_id'].unique())}
    item_df["item_id"] = raw_item_df["item_id"].map(item_id_idx)
    item_title_idx = {v: i for i, v in enumerate(raw_item_df['title'].unique())}
    item_df["title"] = raw_item_df["title"].map(item_title_idx)
    print(item_df)

    raw_data = load_rating_data_file(input_rating_path_file)
    data = pd.DataFrame()
    data['user_id'] = raw_data['user_id'].map(user_id_idx).fillna(-1).astype(int)
    data['item_id'] = raw_data['item_id'].map(item_id_idx).fillna(-1).astype(int)
    data['rating'] = raw_data['rating']
    data = data[data['user_id'] != -1]
    data = data[data['item_id'] != -1]
    print(data)

    # Create training datasets
    np.random.seed(0)
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    train_test_ratio = 0.9
    train_index = arr[:int(len(arr) * train_test_ratio)]
    test_index = arr[int(len(arr) * train_test_ratio):]
    train_set = data.iloc[train_index, :]
    test_set = data.iloc[test_index, :]
    train_set = [tuple(row) for row in train_set.itertuples(index=False, name=None)]
    test_set = [tuple(row) for row in test_set.itertuples(index=False, name=None)]
    return train_set, test_set, user_df, item_df


def evaluation(y_pred, y_true):
    """
    Args:
        y_pred: predicted labels
        y_true: true labels
    Returns:
        precision: precision score
        recall: recall score
        accuracy: accuracy score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


class Deep_wide(nn.Module):
    def __init__(self, feature_num, user_df, item_df, hidden_dim=64):
        """
        Args:
            feature_num: number of features
            user_df: DataFrame with columns: user_id, last_user_agent
            item_df: DataFrame with columns: item_id, title
            hidden_dim: dimension of the embedding
        Returns:
            None
        """
        super(Deep_wide, self).__init__()
        self.features = nn.Embedding(feature_num, hidden_dim, max_norm=1)
        self.user_df = user_df
        self.item_df = item_df
        total_feature_num = user_df.shape[1] + item_df.shape[1]
        self.mlp_layer = self.__mlp(hidden_dim * total_feature_num)
        self.linear_layer = nn.Linear(hidden_dim * total_feature_num, 1)

    def __mlp(self, hidden_dim):
        """
        Args:
            hidden_dim: dimension of the embedding
        Returns:
            torch MLP model
        """
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid())

    def Linear_part(self, feature_embs):
        """
        Args:
            feature_embs: feature embeddings
        Returns:
            torch linear model
        """
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        output = self.linear_layer(feature_embs)
        return torch.squeeze(output)

    def Deep_part(self, feature_embs):
        """
        Args:
            feature_embs: feature embeddings
        Returns:
            torch deep model using MLP
        """
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        output = self.mlp_layer(feature_embs)
        return torch.squeeze(output)

    def concat_user_item_vec(self, u, i):
        """
        Args:
            u: user index
            i: item index
        Returns:
            concat_vec: concatenated user and item vectors
        """
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        concat_vec = torch.cat([users, items], dim=1)
        return concat_vec

    def forward(self, u, i):
        """
        Args:
            u: user index
            i: item index
        Returns:
            out: predicted rating
        """
        concat_vec_index = self.concat_user_item_vec(u, i)
        concat_vec_embs = self.features(concat_vec_index)
        linear_out = self.Linear_part(concat_vec_embs)
        deep_out = self.Deep_part(concat_vec_embs)
        out = torch.sigmoid(linear_out + deep_out)
        return out


def train_and_evaluate(local_click_path_file, local_item_path_file, local_user_path_file):
    train_set, test_set, user_df, item_df = produce_train_data(local_click_path_file, local_item_path_file,
                                                               local_user_path_file)
    features_values_count = len(item_df.values.reshape(-1)) + 1
    model = Deep_wide(features_values_count, user_df, item_df)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    num_epochs = 20
    for epoch in range(num_epochs):
        for user, item, rating in DataLoader(train_set, batch_size=512, shuffle=True):
            optimizer.zero_grad()
            predictions = model(user, item)
            loss = loss_fn(predictions, rating.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        y_pred = np.array([1 if i >= 0.5 else 0 for i in predictions])
        y_true = rating.detach().numpy()
        precision, recall, acc = evaluation(y_pred, y_true)
        print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(precision, recall, acc))

        user_test = torch.tensor(test_set)[:, 0].detach()
        item_test = torch.tensor(test_set)[:, 1].detach()
        predictions = model(user_test, item_test)
        y_pred = np.array([1 if i >= 0.5 else 0 for i in predictions])
        y_true = torch.tensor(test_set)[:, 2].detach().float()
        precision, recall, acc = evaluation(y_pred, y_true)
        print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(precision, recall, acc))
        print('----------------------------------------------------------------------------------------')


if __name__ == "__main__":
    train_and_evaluate(local_click_path_file, local_item_path_file, local_user_path_file)
