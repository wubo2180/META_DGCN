import argparse
import numpy as np
import torch
import random
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
                print "Epoch {}: Loss {}\n".format(e,loss)             
                print 'Updated Model Parameter Theta\n'
                print 'Sampling Next Batch of Tasks \n'
                print '---------------------------------\n'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--data', default='cora', help='dateset')
    parser.add_argument('--dev', type=int, default=0, help='device id')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
    parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    