import argparse
import numpy as np
import torch
import random
import learn2learn as l2l

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split,StaticGraphTemporalSignal
from torch_geometric.utils import negative_sampling
from  utils import *

from dataset import *
import sys
import os
# os.environ["http_proxy"] = "http://127.0.0.1:8081"
# os.environ["https_proxy"] = "http://127.0.0.1:1231"
def sample_points():
    pass
class MAML(object):
    def __init__(self):
        """
        定义参数,实验中用到10-way,10-shot
        """
        # 共有10个任务
        self.num_tasks = 10

        # 每个任务的数据量：10-shot
        self.num_samples = 10

        # 训练的迭代次数
        self.epochs = 10000

        # 内循环中，学习率，用来更新\theta'
        self.alpha = 0.0001

        # 外循环的学习率，用来更新meta模型的\theta
        self.beta = 0.0001

        # meta模型初始化的参数
        self.theta = np.random.normal(size=50).reshape(50, 1)

    # sigmoid函数
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))

    #now let us get to the interesting part i.e training :P
    def train(self):

        # 循环epoch次数
        for e in range(self.epochs):        

            self.theta_ = []

            # 利用support set
            for i in range(self.num_tasks):

                # 抽样k个样本出来，k-shot
                XTrain, YTrain = sample_points(self.num_samples)

                # 前馈神经网络
                a = np.matmul(XTrain, self.theta)
                YHat = self.sigmoid(a)

                # 计算交叉熵loss
                loss = ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 -YTrain.T), np.log(1 - YHat)))/self.num_samples)[0][0]

                # 梯度计算，更新每个任务的theta_，不需要更新meta模型的参数theta
                gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_samples
                self.theta_.append(self.theta - self.alpha*gradient)


            # 初始化meta模型的梯度
            meta_gradient = np.zeros(self.theta.shape)

            # 利用query set
            for i in range(self.num_tasks):

                # 在meta-test阶段，每个任务抽取10个样本出来进行
                XTest, YTest = sample_points(10)

                # 前馈神经网络
                a = np.matmul(XTest, self.theta_[i])
                YPred = self.sigmoid(a)

                # 这里需要叠加每个任务的loss
                meta_gradient += np.matmul(XTest.T, (YPred - YTest)) / self.num_samples


            # 更新meat模型的参数theta
            self.theta = self.theta-self.beta*meta_gradient/self.num_tasks

            if e%1000==0:
                print ("Epoch {}: Loss {}\n".format(e,loss))
                print ('Updated Model Parameter Theta\n')
                print ('Sampling Next Batch of Tasks \n')
                print ('---------------------------------\n')
def compute_loss():
    pass
def train ():
    maml = l2l.algorithms.MAML(model, lr=0.1)
    opt = torch.optim.SGD(maml.parameters(), lr=0.001)
    for iteration in range(args.epochs):
        opt.zero_grad()
        task_model = maml.clone()  # torch.clone() for nn.Modules
        adaptation_loss = compute_loss(task_model)
        task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
        evaluation_loss = compute_loss(task_model)
        evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
        opt.step()
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='epoch number', default=10)
    parser.add_argument('--n_way', type=int, help='n way', default=3)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=24)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--input_dim', type=int, help='input feature dim', default=1)
    parser.add_argument('--hidden_dim', type=int, help='hidden dim', default=64)
    parser.add_argument('--attention_size', type=int, help='dim of attention_size', default=32)
    parser.add_argument("--data_dir", default=None, type=str, required=False, help="The input data dir.")
    parser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")
    parser.add_argument("--task_setup", default='Disjoint', type=str, required=False, help="Select from Disjoint or Shared Setup. For Disjoint-Label, single/multiple graphs are both considered.")
    parser.add_argument("--method", default='G-Meta', type=str, required=False, help="Use G-Meta")
    parser.add_argument('--task_n', type=int, help='task number', default=1)
    parser.add_argument("--task_mode", default='False', type=str, required=False, help="For Evaluating on Tasks")
    parser.add_argument("--val_result_report_steps", default=100, type=int, required=False, help="validation report")
    parser.add_argument("--train_result_report_steps", default=30, type=int, required=False, help="training report")
    parser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    parser.add_argument("--batchsz", default=1000, type=int, required=False, help="batch size")
    parser.add_argument("--link_pred_mode", default='False', type=str, required=False, help="For Link Prediction")
    parser.add_argument("--h", default=2, type=int, required=False, help="neighborhood size")
    parser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    # args = parser.parse_args()
    # parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    # parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
    # parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    # parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
    # parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
    # parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--patience', type=int, default=100, help='Patience')
    # parser.add_argument('--data', default='cora', help='dateset')
    # parser.add_argument('--dev', type=int, default=0, help='device id')
    # parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    # parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    # parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
    # parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    loader = ChickenpoxDatasetLoader()
# loader = EnglandCovidDatasetLoader()
# loader = PedalMeDatasetLoader()
# loader = LocalChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    print(type(dataset))
    # data_preprocessing(dataset)
    for data in dataset:
        print(data.x)
        print(test(data,args))
        print(data.x)
        sys.exit()
    