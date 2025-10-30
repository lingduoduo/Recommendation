import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
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

data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))

d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()

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

class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super(MF, self).__init__()
        self.P = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        self.Q = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(1) + b_u.squeeze() + b_i.squeeze()
        return outputs

def evaluator(net, test_iter, device):
    net.eval()  # Set the network to evaluation mode
    rmse_accumulator = 0
    total_count = 0
    
    with torch.no_grad():  # No gradients needed for evaluation
        for users, items, ratings in test_iter:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            predictions = net(users, items)
            loss = torch.sqrt(mse_loss(predictions, ratings))
            rmse_accumulator += loss.item() * len(users)  # Weighted sum of the RMSE
            total_count += len(users)  # Total number of samples processed
    
    average_rmse = rmse_accumulator / total_count  # Calculate the average RMSE
    return average_rmse
    

def train_recsys_rating(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, device):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        net.train()  # Set the network to training mode
        metric = d2l.Accumulator(3)  # Sum of training loss, number of examples, total time
        for i, (users, items, ratings) in enumerate(train_iter):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device).float()
            optimizer.zero_grad()
            predictions = net(users, items)
            predictions = predictions.float()
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss.item() * len(users), len(users))
        test_rmse = evaluator(net, test_iter, device) if evaluator else float('nan')
        train_loss = metric[0] / metric[1]
        animator.add(epoch + 1, (train_loss, test_rmse))
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Test RMSE: {test_rmse:.3f}')
    print(f'Finished training on {device}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_users, num_items, train_iter, test_iter = split_and_load_ml100k()
net = MF(30, num_users, num_items)
net.to(device)
net.apply(lambda x: nn.init.normal_(x.weight, mean=0, std=0.01) if type(x) == nn.Embedding else None)

# Training Setup
lr, num_epochs, wd = 0.002, 20, 1e-5
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
loss_fn = nn.MSELoss()
# Run Training
train_recsys_rating(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, device)

user_ids = torch.tensor([20], dtype=torch.long).to(device)  # torch.long is equivalent to 'int' in MXNet
item_ids = torch.tensor([30], dtype=torch.long).to(device)

# Get scores by passing the tensors to the model
scores = net(user_ids, item_ids)
print(scores)