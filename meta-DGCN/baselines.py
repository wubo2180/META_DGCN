
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader,WikiMathsDatasetLoader,MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split,StaticGraphTemporalSignal
from torch_geometric.utils import negative_sampling
from  utils import *
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
import numpy as np
from model import RecurrentGCN
def eval(args,model,test_dataset):
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        snapshot = snapshot.to(args.device)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # print(y_hat-snapshot.y)
        # print((y_hat-snapshot.y)**2)
        # ss
        cost = cost + torch.mean((y_hat-snapshot.y.reshape((-1,1)))**2)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))

def train(args, model,train_dataset, optimizer):
    model.train()
    criterion = nn.MSELoss(reduction='mean')
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        snapshot = snapshot.to(args.device)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # cost += loss(y_hat,snapshot.y)
        loss = criterion(y_hat,snapshot.y.reshape((-1,1)))
        cost += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()
    print('train loss:{:.4}'.format(cost / (time+1)))
def main(args):
    if args.dataset == 'EnglandCovid':
        loader =EnglandCovidDatasetLoader()
    elif args.dataset == 'PedalMe':
        loader = PedalMeDatasetLoader()
    elif args.dataset == 'WikiMaths':
        loader = MontevideoBusDatasetLoader()
    elif args.dataset == 'WindmillOutputLarge':
        loader = MontevideoBusDatasetLoader()
    elif args.dataset == 'Chickenpox':
        loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    for time, data in enumerate(dataset):
        args.input_dim, args.num_nodes = data.x.shape[1], data.x.shape[0]
        break
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    model = RecurrentGCN(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(args.epochs):
        print("====epoch " + str(epoch)+'====')
        train(args,model,train_dataset,optimizer)
        eval(args,model,test_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='epoch number', default=50)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-2)
    parser.add_argument('--decay', type=float, help='decay', default=1e-3)
    parser.add_argument('--train_ratio', type=float, help='train_ratio', default=0.8)
    parser.add_argument('--input_dim', type=int, help='input feature dim', default=4)
    parser.add_argument('--hidden_dim', type=int, help='hidden dim', default=32)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.5)
    parser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='Chickenpox', help='dataset.')
    parser.add_argument('--layer_mode', type=str, default='1', help='layer mode.')
    parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    print(args)
    main(args)
