import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
from RectifiedFlow import RectifiedFlow, train_rectified_flow, plot_flow
from MLP import MLP 
import copy 
# 从二维高斯混合分布中生成10000个样本点
D = 10. #类中心之间的距离
M = D+5 
VAR = 0.3 #每个分量的协方差矩阵为VARxI, 每个高斯分布的方差
DOT_SIZE = 4
COMP = 3 #混合模型中包含3个高斯分布分量

initial_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)])) #创建一个类别分布，权重均等，用于表示选择哪一个高斯分量的概率
initial_comp = MultivariateNormal( # 定义每个分量的高斯分布
    torch.tensor([
        [D * np.sqrt(3) / 2., D / 2.], 
        [-D * np.sqrt(3) / 2., D / 2.], 
        [0.0, - D * np.sqrt(3) / 2.]
        ]).float(), 
        VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
initial_model = MixtureSameFamily(initial_mix, initial_comp)
samples_0 = initial_model.sample([10000])

target_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
target_comp = MultivariateNormal(
    torch.tensor([
        [D * np.sqrt(3) / 2., - D / 2.], 
        [-D * np.sqrt(3) / 2., - D / 2.], 
        [0.0, D * np.sqrt(3) / 2.]
        ]).float(), 
        VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([10000])
print('Shape of the samples:', samples_0.shape, samples_1.shape)

#plt.figure(figsize=(4,4))
#plt.xlim(-M,M)
#plt.ylim(-M,M)
#plt.title(r'Samples from $\pi_0$ and $\pi_1$')
#plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_0$')
#plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_1$')
#plt.legend()

#plt.tight_layout()
#plt.savefig("./samples_pi0_pi1.png", dpi=300)
#plt.show()

x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
x_pairs = torch.stack([x_0,x_1], dim=1)
print(x_pairs.shape)

iterations = 10000
batchsize = 2048
input_dim = 2

rectified_flow_1 = RectifiedFlow(model = MLP(input_dim, hidden_num=100), num_steps=100)
optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr = 5e-3)

rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer, x_pairs, batchsize, iterations)
#plt.figure(figsize=(4,4))
#plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
#plt.title('Training Loss Curve')
#plt.savefig("./TrainingLoss.png")

z_10 = samples_0.detach().clone()
traj = rectified_flow_1.sample_ode(z0 = z_10.detach().clone(), N=100)

#plot_flow(traj,name = "1RectfiedFlow.png")
z_11=traj[-1].detach().clone()
z_pairs = torch.stack([z_10,z_11], dim=1)
print(z_pairs.shape)

reflow_iterations = 50000

rectified_flow_2 = RectifiedFlow(model=MLP(input_dim, hidden_num=100), num_steps=100)
rectified_flow_2.net = copy.deepcopy(rectified_flow_1) # we fine-tune the model from 1-Rectified Flow for faster training.
optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=5e-3)

rectified_flow_2, loss_curve = train_rectified_flow(rectified_flow_2, optimizer, z_pairs, batchsize, reflow_iterations)
plt.plot(np.linspace(0, reflow_iterations, reflow_iterations+1), loss_curve[:(reflow_iterations+1)])
plt.savefig("./TrainingLoss2.png")

traj2 = rectified_flow_2.sample_ode(z0 = z_10.detach().clone(), N=100)
plot_flow(traj2, name = "2RectfiedFlow.png")
