import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#超参
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUTE_SIZE = 28
LR = 0.001
DOWNLOAD_MNIST = False   #是否下载数据集

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:200]/255
test_y = test_data.test_labels[:200]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUTE_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_c, h_n) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        output = rnn(batch_x.view(-1, 28, 28))
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y).numpy()/test_y.size(0)
            print(sum(pred_y == test_y).numpy())
            print(test_y.size(0))
            print('Epoch: ', epoch, '| step: ', step,  '| train loss: %.4f' % loss.item(), '|test accuracy: %.4f' % accuracy)


test_output = rnn(test_x[10:20].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[10:20].data.numpy().squeeze(), 'real number')