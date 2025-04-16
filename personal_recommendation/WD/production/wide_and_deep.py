import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# Load the dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'label'
]

train_data = pd.read_csv('train.csv', names=column_names, skipinitialspace=True)
test_data = pd.read_csv('test.csv', names=column_names, skipinitialspace=True)

# Remove rows with missing values
train_data = train_data.replace('?', np.nan).dropna()
test_data = test_data.replace('?', np.nan).dropna()

# Convert label to binary
train_data['label'] = train_data['label'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
test_data['label'] = test_data['label'].apply(lambda x: 1 if x.strip() == '>50K' else 0)


categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]
numerical_cols = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]


class Dataset(Dataset):
    def __init__(self, dataframe, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.labels = dataframe['label'].values.astype(np.float32)
        self.cat_data = dataframe[cat_cols].astype('category')
        self.num_data = dataframe[num_cols].astype(np.float32)

        # Create category encoders
        self.cat_encoders = {}
        for col in cat_cols:
            self.cat_data[col] = self.cat_data[col].cat.codes
            self.cat_encoders[col] = len(self.cat_data[col].unique())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cat_feats = torch.tensor(self.cat_data.iloc[idx].values, dtype=torch.long)
        num_feats = torch.tensor(self.num_data.iloc[idx].values, dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return cat_feats, num_feats, label


class WideAndDeep(nn.Module):
    def __init__(self, embedding_sizes, num_numerical, hidden_units=[64, 32]):
        super(WideAndDeep, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(categories, min(50, (categories + 1) // 2))
            for categories in embedding_sizes
        ])
        emb_dim = sum([emb.embedding_dim for emb in self.embeddings])
        self.bn_numeric = nn.BatchNorm1d(num_numerical)
        self.deep = nn.Sequential(
            nn.Linear(emb_dim + num_numerical, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_units[1], 1)

    def forward(self, x_cat, x_num):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x_num = self.bn_numeric(x_num)
        x = torch.cat([x, x_num], 1)
        x = self.deep(x)
        x = torch.sigmoid(self.output(x))
        return x

# Create datasets
train_dataset = Dataset(train_data, categorical_cols, numerical_cols)
test_dataset = Dataset(test_data, categorical_cols, numerical_cols)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get embedding sizes
embedding_sizes = [train_dataset.cat_encoders[col] for col in categorical_cols]

# Initialize model
model = WideAndDeep(embedding_sizes, len(numerical_cols))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for x_cat, x_num, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x_cat, x_num).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')


# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_cat, x_num, y in test_loader:
        outputs = model(x_cat, x_num).squeeze()
        predicted = (outputs > 0.5).float()
        total += y.size(0)
        correct += (predicted == y).sum().item()
print(f'Accuracy: {100 * correct / total:.2f}%')






