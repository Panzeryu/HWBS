import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt


LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class Net(torch.nn.Module):  # 继承 torch 的 Module,搭建神经网路
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x


net_SGD = Net(n_feature=1, n_hidden=20, n_output=1)
net_Monentum = Net(n_feature=1, n_hidden=20, n_output=1)
net_RMSprop = Net(n_feature=1, n_hidden=20, n_output=1)
net_Adam = Net(n_feature=1, n_hidden=20, n_output=1)
nets = [net_SGD, net_Monentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Monentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_lis = [[], [], [], []]

for epoch in range(EPOCH):
    print(epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, opt, l_lis in zip(nets, optimizers, losses_lis):
            out = net(batch_x)  # 喂给 net 训练数据 x, 输出分析值
            loss = loss_func(out, batch_y)  # 计算两者的误差
            opt.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 误差反向传播, 计算参数更新值
            opt.step()  # 将参数更新值施加到 net 的 parameters 上
            l_lis.append(loss.item())


labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_lis in enumerate(losses_lis):
    plt.plot(l_lis, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()