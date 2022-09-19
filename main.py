import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import learn2learn as l2l
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader,WikiMathsDatasetLoader,PemsBayDatasetLoader,WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split,StaticGraphTemporalSignal
from torch_geometric.utils import negative_sampling
from  utils import *
import torch.optim as optim
from tqdm import tqdm
from model import metaDynamicGCN
from datasets import *
def compute_space_loss(embedding, index_set, criterion_space):
    # embedding = torch.relu(embedding)
    pos_score = torch.sum(embedding[index_set[0]] * embedding[index_set[1]], dim=1)
    neg_score = torch.sum(embedding[index_set[0]] * embedding[index_set[1]], dim=1)
    loss = criterion_space(pos_score, torch.ones_like(pos_score)) + \
           criterion_space(neg_score, torch.zeros_like(neg_score))
    return loss

def compute_temporal_loss(embedding, index_set, snapshot, criterion_temporal):
    
    
    y = snapshot.y[index_set]
    # embedding = torch.sigmoid(embedding[index_set]).reshape(-1)
    embedding = torch.relu(embedding[index_set]).reshape(-1)
    loss = criterion_temporal (embedding, y)
    return loss

def train (args, model, maml, optimizer, train_dataset, criterion_space, criterion_temporal):
    

    cost = 0
    for time, snapshot in enumerate(tqdm(train_dataset,ncols=100)):

        snapshot = snapshot.to(args.device)
        embedding = model(snapshot)
        task_model = maml.clone()
        query_space_loss, query_temporal_loss =0.0,0.0
        
        space_suppport_set = snapshot.pos_sup_edge_index
        space_query_set = snapshot.neg_sup_edge_index
        temporal_suppport_set = snapshot.temporal_que_index
        temporal_query_set = snapshot.temporal_que_index
        
        for i in range(args.update_sapce_step):
            support_space_loss = compute_space_loss(embedding, space_suppport_set, criterion_space)
            # print(support_space_loss)
            task_model.adapt(support_space_loss, allow_unused=True, allow_nograd = True)
            query_space_loss += compute_space_loss(embedding, space_query_set, criterion_space)

        for i in range(args.update_temporal_step):

            suppport_temporal_loss = compute_temporal_loss(embedding,temporal_suppport_set,snapshot,criterion_temporal)
            task_model.adapt(suppport_temporal_loss, allow_unused=True, allow_nograd = True)
            query_temporal_loss += compute_temporal_loss(embedding,temporal_query_set,snapshot,criterion_temporal)

        optimizer.zero_grad()

        evaluation_loss = 0.5*query_space_loss + 0.5*query_temporal_loss
        
        evaluation_loss.backward() 
        optimizer.step()
    print('meta train loss: {:.4f}'.format(evaluation_loss.item()))

def eval(args, model,test_dataset):
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        snapshot = snapshot.to(args.device)
        y_hat = model(snapshot)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))

def main(args):
    if args.dataset == 'EnglandCovid':
        loader = EnglandCovidDatasetLoader()
    elif args.dataset == 'PedalMe':
        loader = PedalMeDatasetLoader()
    elif args.dataset == 'WikiMaths':
        loader = WikiMathsDatasetLoader()

    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)
    train_dataset, args.num_nodes, args.input_dim = data_preprocessing(train_dataset,args)
    # print(node)
    model = metaDynamicGCN(args).to(device)
    maml = l2l.algorithms.MAML(model, lr=args.update_lr)
    optimizer = optim.Adam(maml.parameters(), lr=args.meta_lr, weight_decay=args.decay)
    criterion_space = nn.BCEWithLogitsLoss()
    criterion_temporal = nn.MSELoss(reduction='mean')
    for epoch in range(args.epochs):
        print("====epoch " + str(epoch)+'====')
        train(args, model, maml, optimizer, train_dataset, criterion_space, criterion_temporal)
        eval(args, maml,test_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='epoch number', default=50)
    parser.add_argument('--num_nodes', type=int, help='graph nodes')
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=100)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=100)
    # parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    parser.add_argument('--update_sapce_step', type=int, help='task-level inner update steps', default=1)
    parser.add_argument('--update_temporal_step', type=int, help='update steps for finetunning', default=1)
    parser.add_argument('--decay', type=float, help='decay', default=1e-3)
    parser.add_argument('--train_ratio', type=float, help='train_ratio', default=0.8)
    parser.add_argument('--input_dim', type=int, help='input feature dim', default=4)
    parser.add_argument('--hidden_dim', type=int, help='hidden dim', default=32)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.5)

    parser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='EnglandCovid', help='dataset.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    print(args) ###ddd
    #11111
    print(args) ###dsdasdas
    main(args)
