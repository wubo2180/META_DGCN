import torch.nn as nn
from torch_geometric_temporal.nn import DCRNN
import torch.nn as nn
import torch.functional as F
class metaDynamicGCN(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.encoder = DCRNN(in_channels=args.input_dim, out_channels=args.hidden_dim)
        self.linear = nn.Linear(32, 1)
        self.loss = nn.MSELoss()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self,data,optimizer):
        h = self.encoder(data.x, data.edge_index, data.edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        # for i in range
        return h
    