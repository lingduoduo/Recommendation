import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from sentence_transformers import SentenceTransformer

pd.options.display.max_columns = 100

# Load datasets
df1 = pd.read_csv("../data/ratings.txt", delimiter=",", header=0, names=["user_id", "item_id", "rating", "timestamp"])
df5 = pd.read_csv("../data/movies.txt", delimiter=",", header=0, names=["movieId", "title", "genres"])

# Merge titles
df1 = df1.merge(df5, left_on="item_id", right_on="movieId", how="left")

# Negative sampling
ngSize = 80
df_neg = df1.loc[0:2]
for j in df1["user_id"].unique():
    dft = df1[df1["user_id"] != j].sample(ngSize, random_state=j+1)
    dft["user_id"] = j
    df_neg = pd.concat((df_neg, dft), axis=0)

df_neg = df_neg.iloc[2:]
df_neg["like"] = 0
df1["like"] = 1

# Train/test split
df_train, df_test = train_test_split(df1, train_size=.8, random_state=42, shuffle=True)
df_train = pd.concat((df_train, df_neg), axis=0)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Sentence Transformer for movie title embeddings
model_strans = SentenceTransformer('all-MiniLM-L6-v2')
dict_titlEmb = {title: torch.tensor(model_strans.encode(title)) for title in df5["title"].unique()}


class cls_dataset(Dataset):
    def __init__(self, data):
        self.m_data = data

    def __len__(self):
        return len(self.m_data)

    def __getitem__(self, index):
        row = self.m_data.iloc[index]
        title_embed = dict_titlEmb[row["title"]]
        user_tensor = torch.tensor([row["user_id"]], dtype=torch.float32)
        item_tensor = torch.cat((
            torch.tensor([row["item_id"], row["timestamp"]], dtype=torch.float32),
            title_embed
        ))
        like_tensor = torch.tensor(row["like"], dtype=torch.long)  # CrossEntropy needs int64 class index
        rating_tensor = torch.tensor(row["rating"], dtype=torch.float32)
        return user_tensor, item_tensor, like_tensor, rating_tensor


class cls_model(nn.Module):
    def __init__(self, userCount, itemCount, user_embSize=16, item_embSize=16):
        super().__init__()
        self.m_userEmb = nn.Embedding(userCount + 1, user_embSize)
        self.m_itemEmb = nn.Embedding(itemCount + 1, item_embSize)

        self.m_modelUser = nn.Sequential(
            nn.Linear(user_embSize, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.m_modelItem = nn.Sequential(
            nn.Linear(item_embSize + 1 + 384, 128),  # +384 for sentence embed
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        self.m_modelClassify = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, dataUser, dataItem):
        embu = self.m_userEmb(dataUser[:, 0].long())
        embi = self.m_itemEmb(dataItem[:, 0].long())
        inputU = torch.cat((embu, dataUser[:, 1:]), dim=1)
        inputI = torch.cat((embi, dataItem[:, 1:]), dim=1)
        logitsU = self.m_modelUser(inputU)
        logitsI = self.m_modelItem(inputI)
        logits = self.m_modelClassify(torch.cat((logitsU, logitsI), dim=1))
        return logits

    def predict(self, userID):
        embu = self.m_userEmb(userID.long())
        embi = self.m_itemEmb.weight.data
        res = embu @ embi.T
        normU = torch.linalg.norm(embu, dim=1).unsqueeze(1)
        normI = torch.linalg.norm(embi, dim=1).unsqueeze(0)
        return res / (normU @ normI)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

modelRec = cls_model(userCount=df1["user_id"].max(), itemCount=df1["item_id"].max())
modelRec.to(device)

train_dataset = cls_dataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelRec.parameters(), lr=5e-3)

# Training loop
epochs = 1
modelRec.train()
for epoch in range(epochs):
    total_loss = 0
    for i, (x_user, x_item, y, _) in enumerate(train_loader):
        x_user, x_item, y = x_user.to(device), x_item.to(device), y.to(device)
        logits = modelRec(x_user, x_item)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {total_loss / (i + 1):.4f}")


class cls_modelRank(nn.Module):
    def __init__(self, userEmb, itemEmb):
        super().__init__()

        self.m_userEmb = userEmb
        self.m_itemEmb = itemEmb

        self.m_modelClassify = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
            )

    def forward(self, dataUser, dataItem):
        embu = self.m_userEmb.forward(dataUser.long())
        embi = self.m_itemEmb.forward(dataItem.long())


        logits = self.m_modelClassify.forward( torch.cat((embu, embi), dim=1) )
        return logits

    def f_getUserEmb(self, ids):
        assert type(ids) == torch.Tensor
        return self.m_userEmb.forward(ids)


epochs = 1

ds_trainLoader = DataLoader(torch.tensor(df_train[["user_id", "item_id", "rating"]].to_numpy(), dtype=torch.float32),
                            batch_size=1000)
ds_testLoader = DataLoader(torch.tensor(df_test[["user_id", "item_id", "rating"]].to_numpy(), dtype=torch.float32),
                           batch_size=1000)

modelRank = cls_modelRank(modelRec.m_userEmb, modelRec.m_itemEmb).to(device)

criterion = nn.MSELoss()
optim = torch.optim.Adam(modelRank.parameters(), lr=5e-3)

modelRank.train()

for j in range(epochs):
    loss_acc = 0
    for i, x in enumerate(ds_trainLoader):
        x = x.to(device)

        logits = modelRank.forward(x[:, 0], x[:, 1])

        loss = criterion.forward(logits, x[:, 2].squeeze())

        optim.zero_grad()

        loss.backward()

        optim.step()

        loss_acc += loss.item()
        if i % 10 == 0:
            print(f"Train train MSE loss {loss_acc / 10}")
            loss_acc = 0

            modelRank.eval()

            total_acc = 0.0
            for k, batch in enumerate(ds_testLoader):
                batch = batch.to(device)

                y_pred = modelRank.forward(batch[:, 0], batch[:, 1])

                lossMSE = criterion.forward(batch[:, 2], y_pred)

                total_acc += lossMSE.item()
            print(f"Total val MSE loss: {total_acc / (k + 1)}")
            modelRec.train()

import faiss

index = faiss.IndexFlatL2(modelRank.m_itemEmb.embedding_dim)
index.add(modelRank.m_itemEmb.weight.data.detach().cpu().numpy())
print(index.ntotal)
q = modelRank.f_getUserEmb(torch.tensor([1,2,10]).to(device))
q = q.detach().cpu().numpy()
print(index.search(q,5))