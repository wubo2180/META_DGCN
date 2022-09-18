import torch.nn as nn
from torch_geometric_temporal.nn import DCRNN,GConvGRU,GConvLSTM,EvolveGCNH,EvolveGCNO,TGCN,A3TGCN,STConv
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from layers import *
class metaDynamicGCN(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.encoder = GCNGRU(args.input_dim,args.hidden_dim)
        self.linear = nn.Linear(args.hidden_dim, 1)
        self.dropout = nn.Dropout(p=args.dropout)
        self.relu = nn.ReLU()
    def forward(self,data):
        h = self.encoder(data.x, data.edge_index, data.edge_weight)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.linear(h)
        return h


