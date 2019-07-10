import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#超参
EPOCH = 60
TIME_STEP = 10
INPUTE_SIZE = 1
LR = 0.001


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUTE_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size/hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None
# plt.figure(1, figsize=(12, 5))
# plt.ion()

for step in range(EPOCH):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    plt.cla()
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-', label='output(sin)')
    plt.plot(steps, y_np.flatten().flatten(), 'r-', label='target(cos)')
    plt.pause(0.1)
    # plt.plot(steps, y_np.flatten(), 'r-')
    # plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    # plt.draw()
    # plt.pause(0.05)


plt.ioff()
plt.show()


