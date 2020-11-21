import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = [64, 64]

        conv0 = nn.Conv2d(1, self.layers[0], kernel_size=3, stride=2)
        self.conv0 = nn.utils.spectral_norm(conv0)
        conv1 = nn.Conv2d(self.layers[0], self.layers[1], kernel_size=3, stride=2)
        self.conv1 = nn.utils.spectral_norm(conv1)

        self.fc = nn.Linear(self.layers[1] * 6 * 6, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)

        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc(x))

        return x