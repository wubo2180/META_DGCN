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
##ss
class RecurrentGCN(nn.Module):
    def __init__(self, args):
        super(RecurrentGCN, self).__init__()
        self.mode = False
        self.RNNGCN = ''
        if args.layer_mode == '0':
            self.recurrent = GCNConv(args.input_dim, args.hidden_dim)
        if args.layer_mode == '1':
            self.recurrent = GConvGRU(args.input_dim, args.hidden_dim, 2)
        elif args.layer_mode == '2':
            self.recurrent = DCRNN(args.input_dim, args.hidden_dim, 1)
        elif args.layer_mode == '3':
            self.recurrent = GConvLSTM(args.input_dim, args.hidden_dim, 1)
            self.RNNGCN = 'GConvLSTM'
        elif args.layer_mode == '4':
            self.recurrent = EvolveGCNH(args.num_nodes, args.input_dim)
            self.linear_ = nn.Linear(args.input_dim,args.hidden_dim)
            self.mode = True
        elif args.layer_mode == '5':
            self.recurrent = EvolveGCNO(args.input_dim)
            self.linear_ = nn.Linear(args.input_dim,args.hidden_dim)
            self.mode = True
        elif args.layer_mode == '6':
            self.recurrent = TGCN(args.input_dim, args.hidden_dim)
        elif args.layer_mode == '7':
            self.recurrent = A3TGCN(args.input_dim, args.hidden_dim,4)
        elif args.layer_mode == '8':
            self.recurrent = STConv(args.num_nodes,args.input_dim, args.hidden_dim,args.hidden_dim,1,1)
        self.linear = nn.Linear(args.hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        if self.RNNGCN == 'GConvLSTM':
            h, c = self.recurrent(x, edge_index, edge_weight)
        else:
            h = self.recurrent(x, edge_index, edge_weight)
        if self.mode == True:
            h = self.linear_(h)
        h = F.relu(h)
        h = self.linear(h)
        return h
