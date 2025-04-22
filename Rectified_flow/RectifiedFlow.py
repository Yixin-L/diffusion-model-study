import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class RectifiedFlow():
    def __init__(self, model = None, num_steps = 1000):
        self.model = model
        self.N = num_steps

    def get_train_tuple(self, z0=None, z1=None):
        t = torch.rand(z1.shape[0],1) # 随机采样时间 t ∈ [0,1]
        z_t = t * z1 + (1. -t)*z0 # zt是z0和z1的插值
        target = z1-z0

        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0=None, N = None):
        if N is None:
            N = self.N
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.detach().clone() #不包含梯度信息
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1)) * i/N
            pred = self.model(z,t)
            z = z.detach().clone() + pred * dt #显示Euler更新

            traj.append(z.detach().clone())
        
        #plot_flow(traj)
        return traj
    
def plot_flow(traj = None, name = None):
    plt.figure()
    mycolor = ['b', 'r']
    color_candidate = ['g', 'c', 'm', 'y', 'k']

    sample_num = 20
    this_step = 5
    for i in range(sample_num): #选择了20个样本，作者总共设置了1万个样本
        pointindex = list(range(0, (len(traj) - 1), this_step)) + [len(traj) - 1] #作者设置的每条线101个点，但画多了看不清，少画些吧
        thiscolor = random.sample(color_candidate, 1)[0]
        for k in range(len(pointindex) - 1):
            plt.plot([traj[pointindex[k]][i, 0], traj[pointindex[k + 1]][i, 0]], [traj[pointindex[k]][i, 1], traj[pointindex[k + 1]][i, 1]], c=thiscolor) #画这一小段的线，使用本次随机到的颜色
            if(k == 0): #如果是起点，点就画大些，使用指定颜色蓝色
                plt.scatter([traj[pointindex[k]][i, 0]], [traj[pointindex[k]][i, 1]], s = 30, c=mycolor[0])
            elif(k + 1 == len(pointindex) - 1): #如果是终点，点就画大些，使用指定颜色红色
                plt.scatter([traj[pointindex[k + 1]][i, 0]], [traj[pointindex[k + 1]][i, 1]], s = 30, c=mycolor[1])
            else: #那就剩中间点了，用小尺寸，使用本次随机到的颜色
                plt.scatter([traj[pointindex[k]][i, 0]], [traj[pointindex[k]][i, 1]], s = 10, c=thiscolor)
                plt.scatter([traj[pointindex[k + 1]][i, 0]], [traj[pointindex[k + 1]][i, 1]], s = 10, c=thiscolor)

    plt.savefig(f"./{name}")
    plt.show()

    
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    """
    pairs训练样本对 [N,2,D], 表示(z0,z1)对
    batchsize: 每一轮训练使用的样本数
    """
    loss_curve = []
    for i in range(inner_iters+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize] #随机达伦索引后取前batchsize个
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)

        pred = rectified_flow.model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        loss_curve.append(np.log(loss.item())) ## to store the loss curve

    return rectified_flow, loss_curve