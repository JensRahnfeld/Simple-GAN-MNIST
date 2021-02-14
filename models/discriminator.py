import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_layers import _Conv2d


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = [128, 256, 512]

        self.conv0 = _Conv2d(1, self.layers[0], kernel_size=6, stride=2,
                             norm="none", activation="leaky_relu")
        self.conv1 = _Conv2d(self.layers[0], self.layers[1], kernel_size=4, stride=2, padding=1,
                             norm="spectral", activation="leaky_relu")
        self.conv2 = _Conv2d(self.layers[1], self.layers[2], kernel_size=4, stride=2, padding=1,
                             norm="spectral", activation="leaky_relu")

        self.fc = nn.Linear(self.layers[2] * 3 * 3, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, start_dim=1)
        logit = self.fc(x)

        return logit