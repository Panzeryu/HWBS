import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False   #是否下载数据集

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0], cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

test_data = torchvision.datasets.MNIST(root='./mnist', train=False,)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:200]/255
test_y = test_data.test_labels[:200]

train_datafft = torchvision.datasets.ImageFolder(root='F:/HWBS/picture', transform=torchvision.transforms.ToTensor(),)
train_loaderfft = Data.DataLoader(dataset=train_datafft, batch_size=BATCH_SIZE, shuffle=True,)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  #图像通道数
                out_channels=4, #卷积核数/输出通道数
                kernel_size=9,  #卷积核大小
                stride=1,       #步长
                padding=0,      #外围填充
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.Linear(),
            # nn.Linear(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,  # 图像通道数
                out_channels=16,  # 卷积核数/输出通道数
                kernel_size=11,  # 卷积核大小
                stride=1,  # 步长
                padding=0,  # 外围填充
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.Linear(),
            # nn.Linear(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 图像通道数
                out_channels=64,  # 卷积核数/输出通道数
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 外围填充
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.Linear(),
            # nn.Linear(),
        )
        self.out0 = nn.AdaptiveAvgPool2d(1)
        self.out1 = nn.Linear(64 * 1 * 1, 10)
        self.out2 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out0(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        # output = x
        output = self.out2(x)
        return output


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
list_x = []
list_loss = []

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        list_x.append(step)
        list_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y).numpy()/test_y.size(0)
            print('Epoch: ', epoch, '| step: ', step,  '| train loss: %.4f' % loss.item(), '|test accuracy: %.4f' % accuracy)


test_output = cnn(test_x[:100])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:100].data.numpy().squeeze(), 'real number')


