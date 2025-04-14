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
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score,accuracy_score
import torch.optim as optim

data = pd.read_csv("../data/search_click.csv")
user_ids = data["user_id"].unique()
user_2_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

item_df = pd.read_csv("../data/item_desc.csv")
item_ids = item_df["item_id"].unique()
item_2_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

print(user_2_idx)
print(item_2_idx)

# class ResidualUnit(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ResidualUnit, self).__init__()
#         self.fc1 = nn.Linear(input_size, output_size)
#         self.fc2 = nn.Linear(output_size, output_size)
#         self.downsample = nn.Linear(input_size, output_size) if input_size != output_size else None
#
#     def forward(self, x):
#         identity = x
#
#         out = F.relu(self.fc1(x))
#         out = self.fc2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = F.relu(out)
#
#         return out
#
# class deep_crossing(nn.Module):
#     def __init__(self, num_features, factor_dim, res_layers_parameter_list, hidden_dims, user_df, item_df):
#         super(deep_crossing, self).__init__()
#
#         self.user_df = user_df
#         self.item_df = item_df
#
#         self.embeddings = nn.Embedding(num_features, factor_dim)
#
#         self.res_layers = nn.Sequential()
#         for i in range(len(res_layers_parameter_list)):
#             input_size = factor_dim * (len(user_df) + len(item_df)) if i == 0 else res_layers_parameter_list[i - 1]
#             output_size = res_layers_parameter_list[i]
#             res_unit = ResidualUnit(input_size, output_size)
#             self.res_layers.add_module(f"residual_unit_{i}", res_unit)
#
#         self.predict_layer = nn.Linear(res_layers_parameter_list[-1], 1)
#
#
#     def concat_user_item_vec( self, u, i ):
#           user_ids = u.tolist()
#           users = torch.LongTensor( self.user_df.loc[user_ids].values)
#           item_ids = i.tolist()
#           items = torch.LongTensor( self.item_df.loc[item_ids].values )
#           concat_vec = torch.cat( [ users, items ], dim = 1 )
#           return concat_vec
#
#     def forward(self, u, i):
#         concat_vec_index = self.concat_user_item_vec( u, i )
#         embeddings = self.embeddings( concat_vec_index )
#         embeddings = embeddings.reshape(concat_vec_index.shape[0], -1)
#         res_output = self.res_layers(embeddings)
#         output = self.predict_layer(res_output)
#         return torch.squeeze(torch.sigmoid(output))

# create training and test datasets
np.random.seed(0)
arr = np.arange(data.shape[0])
np.random.shuffle(arr)
train_test_ratio = 0.9
train_index = arr[:int(len(arr)*train_test_ratio)]
test_index = arr[int(len(arr)*train_test_ratio):]
train_set = data.iloc[train_index,:]
test_set = data.iloc[test_index, :]

train_set = [tuple(row) for row in train_set.itertuples(index=False, name=None)]
test_set = [tuple(row) for row in test_set.itertuples(index=False, name=None)]
# print(len(train_set), len(test_set))


num_features = 128
factor_dim = 64
num_layers = 1
hidden_dims = [64]
res_layers_parameter_list = [32,16]

# model = deep_crossing(num_features, factor_dim, res_layers_parameter_list, hidden_dims, user_df, item_df)

# def evaluation(y_pred, y_true):
#     p = precision_score(y_true, y_pred)
#     r = recall_score(y_true, y_pred)
#     acc = accuracy_score(y_true, y_pred)
#     return p, r, acc
#
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# loss_fn = nn.BCELoss()
#
# num_epochs = 10
# for epoch in range(num_epochs):
#   for user, item, rating in DataLoader(train_set, batch_size=512, shuffle=True):
#     optimizer.zero_grad()
#     predictions = model(user, item)
#     loss = loss_fn(predictions, rating.float())
#     loss.backward()
#     optimizer.step()
#   print(f"Epoch {epoch}, Loss: {loss.item()}")
#
#   y_pred = np.array([1 if i >= 0.5 else 0 for i in predictions])
#   y_true = rating.detach().numpy()
#   precision, recall, acc = evaluation(y_pred, y_true)
#   print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(precision, recall, acc))
#
#   user_test = torch.tensor(test_set)[:,0].detach()
#   item_test = torch.tensor(test_set)[:,1].detach()
#   predictions = model(user_test, item_test)
#   y_pred = np.array([1 if i >= 0.5 else 0 for i in predictions])
#   y_true = torch.tensor(test_set)[:,2].detach().float()
#   precision, recall, acc = evaluation(y_pred, y_true)
#   print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(precision, recall, acc))
#   print('----------------------------------------------------------------------------------------')
