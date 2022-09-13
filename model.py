import torch.nn as nn
from torch_geometric_temporal.nn import DCRNN
import torch.nn as nn
import torch.functional as F
class metaDynamicGCN(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.encoder = DCRNN(in_channels=args.input_dim, out_channels=args.hidden_dim,K=1)
        self.linear = nn.Linear(args.hidden_dim, 1)
        self.dropout = nn.Dropout(p=args.dropout)
        self.relu = nn.ReLU()
        # self.loss = nn.MSELoss()
        # self.loss = nn.BCEWithLogitsLoss()
    def forward(self,data):
        h = self.encoder(data.x, data.edge_index, data.edge_weight)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.linear(h)
        # for i in range
        return h
    