import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH = 1
BATCH_SIZE = 100
LR = 0.005
DOWNLOAD_MNIST = False   #是否下载数据集

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0], cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:1000]/255
test_y = test_data.test_labels[:1000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = sum(pred_y == test_y.numpy())/test_y.size(0)
            print('Epoch: ', epoch, '| step: ', step,  '| train loss: %.4f' % loss.item(), '|test accuracy: %.4f' % accuracy)
        if step / 150 == 1:
            break


test_output = cnn(test_x[:100])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:100].data.numpy().squeeze(), 'real number')

Acc = 0
totalnum = 0
for i, j in zip(pred_y, test_y[:10000]):
    totalnum = totalnum + 1
    if i == j.tolist():
        Acc = Acc + 1

print('accuracy=%f',Acc/totalnum)
