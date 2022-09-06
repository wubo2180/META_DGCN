
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split,StaticGraphTemporalSignal
from torch_geometric.utils import negative_sampling
from  utils import *

from dataset import *
import sys
import os
from tqdm import tqdm
import torch.nn as nn
# os.environ["http_proxy"] = "http://127.0.0.1:8081"
# os.environ["https_proxy"] = "http://127.0.0.1:1231"
loader = ChickenpoxDatasetLoader()
# loader = EnglandCovidDatasetLoader()
# loader = PedalMeDatasetLoader()
# loader = LocalChickenpoxDatasetLoader()
dataset = loader.get_dataset()
# print(type(dataset))
# data_preprocessing(dataset)
# for data in dataset:
#     test(data)
#     sys.exit()
# sys.exit()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecurrentGCN(node_features = 4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
loss = nn.MSELoss()
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        snapshot = snapshot.to(device)
        # print(type(snapshot))
        # print(negative_sampling(snapshot.edge_index))
        # sys.exit()
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost += loss(y_hat,snapshot.y)
        # cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))