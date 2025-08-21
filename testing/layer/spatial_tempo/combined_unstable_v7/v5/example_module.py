# example_module.py: Example module integrating CustomConvLayer with standard PyTorch layers

import torch
import torch.nn as nn
import torch.optim as optim

from test import CustomConvLayer  # Import from test1.py

class ExampleNet(nn.Module):
    def __init__(self, kof_path, tprl_path):
        super().__init__()
        self.custom1 = CustomConvLayer(kof_path, tprl_path, in_channels=1, out_channels=4, padding=1, stride=1, kernel_idx=0, verbose=True)
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1)
        self.custom2 = CustomConvLayer(kof_path, tprl_path=None, in_channels=8, out_channels=1, padding=1, stride=1, kernel_idx=1, verbose=True)
        self.relu = nn.ReLU()

    def forward(self, x, epoch=0):
        x = self.custom1(x, epoch=epoch)
        x = self.relu(x)
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.custom2(x, epoch=epoch)
        return x

if __name__ == "__main__":
    net = ExampleNet("test.kof", "full_featured.tprl")
    optimizer = optim.SGD(net.parameters(), lr=0.005, weight_decay=0.001)

    input_data = torch.rand(1, 1, 30, 20, 10)
    target = torch.rand(1, 1, 30, 20, 10)

    epochs = 10
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = net(input_data, epoch=epoch)
        loss = nn.MSELoss()(output, target)
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}, stopping training")
            break
        loss.backward()
        net.custom1.update_histories()
        net.custom2.update_histories()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
