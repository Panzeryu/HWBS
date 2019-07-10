import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import datetime


EPOCH = 2
BATCH_SIZE = 50
LR = 0.002
DOWNLOAD_MNIST = False   #是否下载数据集


def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0], cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

test_data = torchvision.datasets.MNIST(root='./mnist', train=False,)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:200]/255
test_y = test_data.test_labels[:200]


# train_datafft = torchvision.datasets.ImageFolder(root='F:/HWBS/picture', transform=torchvision.transforms.ToTensor(),)   #注意修改地址
# train_loaderfft = Data.DataLoader(dataset=train_datafft, batch_size=BATCH_SIZE, shuffle=True,)
#
# test_datafft = torchvision.datasets.ImageFolder(root='F:/HWBS/picture', transform=torchvision.transforms.ToTensor(),)   #注意修改地址
# test_x = torch.unsqueeze(test_datafft.test_data, dim=1).type(torch.FloatTensor)[:200]/255
# test_y = test_datafft.test_labels[:200]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  #图像通道数
                out_channels=36, #卷积核数/输出通道数
                kernel_size=5,  #卷积核大小
                stride=1,       #步长
                padding=2,      #外围填充
            ),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # nn.Linear(),
            # nn.Linear(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,  # 图像通道数
                out_channels=36,  # 卷积核数/输出通道数
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 外围填充
            ),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # nn.Linear(),
            # nn.Linear(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=36,  # 图像通道数
                out_channels=36,  # 卷积核数/输出通道数
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 外围填充
            ),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=10,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # nn.Linear(),
            # nn.Linear(),
        )
        self.out0 = nn.AdaptiveAvgPool2d(1)
        self.out1 = nn.Linear(128 * 1 * 1, 10)
        self.out2 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out0(x)
        # x = x.view(x.size(0), -1)
        # x = self.out1(x)
        x = torch.squeeze(x)
        # output = x
        output = self.out2(x)
        return output


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
list_x1 = []
list_x2 = []
list_accu = []
list_loss = []
startdate = datetime.datetime.now()  # 获取当前时间
startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式


# accumulation_steps = 20
# evaluation_steps = 100
#
# cnn.zero_grad()  # Reset gradients tensors
# for i, (inputs, labels) in enumerate(train_loader):
#     print(i)
#     predictions = cnn(inputs) # Forward pass
#     loss = loss_func(predictions, labels)  # Compute loss function
#     loss = loss / accumulation_steps # Normalize our loss (if averaged)
#     loss.backward()  # Backward pass
#     if (i+1) % accumulation_steps == 0:  # Wait for several backward steps
#         optimizer.step()  # Now we can do an optimizer step
#         cnn.zero_grad()  # Reset gradients tensors
#     if (i+1) % evaluation_steps == 0:  # Evaluate the model when we...
#         cnn.eval()  # ...have no gradients accumulated
#         test_output = cnn(test_x)
#         pred_y = torch.max(test_output, 1)[1].data.squeeze()
#         accuracy = sum(pred_y == test_y).numpy()/test_y.size(0)
#         print('step: ', i,  '| train loss: %.4f' % loss.item(), '|test accuracy: %.4f' % accuracy)
#         cnn.train()
#         del pred_y


for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        list_x1.append(step)
        list_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y).numpy()/test_y.size(0)
            list_x2.append(step)
            list_accu.append(accuracy)
            print('Epoch: ', epoch, '| step: ', step,  '| train loss: %.4f' % loss.item(), '|test accuracy: %.4f' % accuracy)
            del pred_y


test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y.data.numpy().squeeze(), 'real number')

enddate = datetime.datetime.now()  # 获取当前时间
enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式
print('start date ', startdate)
print('end date ', enddate)
print('Time ', subtime(startdate, enddate)) # enddate > startdate

picx1 = np.array(list_x1)
picx2 = np.array(list_x2)
picy1 = np.array(list_loss)
picy2 = np.array(list_accu)
f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, sharey=False)
ax1.plot(picx1, picy1, 'r-', lw=3)
ax2.plot(picx2, picy2, 'b-', lw=3)
plt.show()


