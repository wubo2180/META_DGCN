import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import learn2learn as l2l
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader,WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split,StaticGraphTemporalSignal
from torch_geometric.utils import negative_sampling
from  utils import *
import torch.optim as optim
from dataset import *
import sys
import os
from tqdm import tqdm
from model import metaDynamicGCN
# os.environ["http_proxy"] = "http://127.0.0.1:8081"
# os.environ["https_proxy"] = "http://127.0.0.1:1231"


def compute_space_loss(embedding, index_set, criterion_space):
    
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
        
        evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
        optimizer.step()
    print(evaluation_loss)

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
    
    if args.dataset == 'Chickenpox':
        loader = ChickenpoxDatasetLoader()
    elif args.dataset == 'EnglandCovid':
        loader = EnglandCovidDatasetLoader()
    elif args.dataset == 'PedalMeDataset':
        loader = PedalMeDatasetLoader()
    elif args.dataset == 'WikiMathsDataset':
        loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)
    train_dataset, args.input_dim = data_preprocessing(train_dataset,args)
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
    parser.add_argument('--n_way', type=int, help='n way', default=3)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=60)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=60)
    # parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    parser.add_argument('--update_sapce_step', type=int, help='task-level inner update steps', default=1)
    parser.add_argument('--update_temporal_step', type=int, help='update steps for finetunning', default=1)
    parser.add_argument('--decay', type=float, help='decay', default=1e-3)
    parser.add_argument('--train_ratio', type=float, help='train_ratio', default=0.8)
    parser.add_argument('--input_dim', type=int, help='input feature dim', default=4)
    parser.add_argument('--hidden_dim', type=int, help='hidden dim', default=32)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.5)

    parser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='WikiMathsDataset', help='dataset.')
    parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    main(args)
# parser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")
# parser.add_argument('--task_n', type=int, help='task number', default=1)
# parser.add_argument("--task_mode", default='False', type=str, required=False, help="For Evaluating on Tasks")
# parser.add_argument("--val_result_report_steps", default=100, type=int, required=False, help="validation report")
# parser.add_argument("--train_result_report_steps", default=30, type=int, required=False, help="training report")
# parser.add_argument("--batchsz", default=1000, type=int, required=False, help="batch size")
# parser.add_argument("--link_pred_mode", default='False', type=str, required=False, help="For Link Prediction")
# parser.add_argument("--h", default=2, type=int, required=False, help="neighborhood size")
# parser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)
# def sample_points():
#     pass
# class MAML(object):
#     def __init__(self):
#         """
#         定义参数,实验中用到10-way,10-shot
#         """
#         # 共有10个任务
#         self.num_tasks = 10

#         # 每个任务的数据量：10-shot
#         self.num_samples = 10

#         # 训练的迭代次数
#         self.epochs = 10000

#         # 内循环中，学习率，用来更新\theta'
#         self.alpha = 0.0001

#         # 外循环的学习率，用来更新meta模型的\theta
#         self.beta = 0.0001

#         # meta模型初始化的参数
#         self.theta = np.random.normal(size=50).reshape(50, 1)

#     # sigmoid函数
#     def sigmoid(self,a):
#         return 1.0 / (1 + np.exp(-a))

#     #now let us get to the interesting part i.e training :P
#     def train(self):

#         # 循环epoch次数
#         for e in range(self.epochs):        

#             self.theta_ = []

#             # 利用support set
#             for i in range(self.num_tasks):

#                 # 抽样k个样本出来，k-shot
#                 XTrain, YTrain = sample_points(self.num_samples)

#                 # 前馈神经网络
#                 a = np.matmul(XTrain, self.theta)
#                 YHat = self.sigmoid(a)

#                 # 计算交叉熵loss
#                 loss = ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 -YTrain.T), np.log(1 - YHat)))/self.num_samples)[0][0]

#                 # 梯度计算，更新每个任务的theta_，不需要更新meta模型的参数theta
#                 gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_samples
#                 self.theta_.append(self.theta - self.alpha*gradient)


#             # 初始化meta模型的梯度
#             meta_gradient = np.zeros(self.theta.shape)

#             # 利用query set
#             for i in range(self.num_tasks):

#                 # 在meta-test阶段，每个任务抽取10个样本出来进行
#                 XTest, YTest = sample_points(10)

#                 # 前馈神经网络
#                 a = np.matmul(XTest, self.theta_[i])
#                 YPred = self.sigmoid(a)

#                 # 这里需要叠加每个任务的loss
#                 meta_gradient += np.matmul(XTest.T, (YPred - YTest)) / self.num_samples


#             # 更新meat模型的参数theta
#             self.theta = self.theta-self.beta*meta_gradient/self.num_tasks

#             if e%1000==0:
#                 print ("Epoch {}: Loss {}\n".format(e,loss))
#                 print ('Updated Model Parameter Theta\n')
#                 print ('Sampling Next Batch of Tasks \n')
#                 print ('---------------------------------\n')