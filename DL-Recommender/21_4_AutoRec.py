import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from d2l import torch as d2l

import os
import pandas as pd
import numpy as np


d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter

def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    
    # Convert lists to PyTorch tensors
    train_u, train_i, train_r = map(torch.tensor, (train_u, train_i, train_r))
    test_u, test_i, test_r = map(torch.tensor, (test_u, test_i, test_r))

    # Create TensorDataset objects
    train_set = TensorDataset(train_u, train_i, train_r)
    test_set = TensorDataset(test_u, test_i, test_r)

    # Create DataLoader objects
    train_iter = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_iter = DataLoader(test_set, batch_size=batch_size)
    
    return num_users, num_items, train_iter, test_iter

class AutoRec(nn.Module):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_users, num_hidden)  # Assuming the input dimension is num_users
        self.decoder = nn.Linear(num_hidden, num_users)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = input.float()  # Ensure input is float for processing
        hidden = F.sigmoid(self.encoder(self.dropout(input)))  # Apply dropout before encoding
        pred = self.decoder(hidden)
        if self.training:  # Check if the model is in training mode
            return pred * torch.sign(input)
        else:
            return pred
        
def evaluator(network, inter_matrix, test_data, device):
    network.eval()  # Ensure the network is in evaluation mode
    scores = []
    
    # Assuming inter_matrix is a list of tensors
    for values in inter_matrix:
        values = values.to(device)  # Move the data to the specified device
        scores.append(network(values).detach().cpu().numpy())  # Compute and collect scores
    
    # Flatten the list of scores and convert it to a numpy array
    recons = np.concatenate(scores, axis=0)
    
    # Calculate the test RMSE
    test_data = test_data.numpy() if isinstance(test_data, torch.Tensor) else test_data
    signed_test_data = np.sign(test_data) * recons
    mse = np.mean((test_data - signed_test_data) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def train_recsys_rating(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, device):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        net.train()  # Set the network to training mode
        metric = d2l.Accumulator(3)  # Sum of training loss, number of examples, total time
        for i, data in enumerate(train_iter):
            optimizer.zero_grad()
            predictions = net(data)
            predictions = predictions.float()
            loss = loss_fn(predictions, data)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss.item() * len(users), len(users))
        test_rmse = evaluator(net, test_iter, device) if evaluator else float('nan')
        train_loss = metric[0] / metric[1]
        animator.add(epoch + 1, (train_loss, test_rmse))
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Test RMSE: {test_rmse:.3f}')
    print(f'Finished training on {device}')


# Load the MovieLens 100K dataset
df, num_users, num_items = read_data_ml100k()
num_users, num_items, train_iter, test_iter = split_and_load_ml100k()
# train_inter_mat = torch.tensor(load_data_ml100k(train_data, num_users, num_items)[3], dtype=torch.float32)
# test_inter_mat = torch.tensor(load_data_ml100k(test_data, num_users, num_items)[3], dtype=torch.float32)

# # DataLoaders
# train_iter = DataLoader(train_inter_mat, batch_size=256, shuffle=True, num_workers=4)
# test_iter = DataLoader(test_inter_mat, batch_size=1024, shuffle=False, num_workers=4)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AutoRec(500, num_users).to(device)
net.apply(lambda m: nn.init.normal_(m.weight, mean=0, std=0.01) if hasattr(m, 'weight') else None)

# Optimizer and Loss
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# Run Training

train_recsys_rating(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, device)

user_ids = torch.tensor([20], dtype=torch.long).to(device)  # torch.long is equivalent to 'int' in MXNet
item_ids = torch.tensor([30], dtype=torch.long).to(device)

# Get scores by passing the tensors to the model
scores = net(user_ids, item_ids)
print(scores)

# Load the MovieLens 100K dataset
user_ids = torch.tensor([20], dtype=torch.long).to(device)  # torch.long is equivalent to 'int' in MXNet
item_ids = torch.tensor([30], dtype=torch.long).to(device)

# Get scores by passing the tensors to the model
scores = net(user_ids, item_ids)
print(scores)