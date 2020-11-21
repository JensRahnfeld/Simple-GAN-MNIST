import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim_latent=100):
        super(Generator, self).__init__()
        
        self.dim_latent = dim_latent
        self.layers = [128, 128, 128]
        
        self.fc = nn.Linear(dim_latent, self.layers[0] * 7 * 7)
        self.deconv0 = nn.ConvTranspose2d(self.layers[0], self.layers[1],
                                            kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(self.layers[1], self.layers[2],
                                            kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(self.layers[2], 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.fc(x))
        x = x.view(batch_size, self.layers[0], 7, 7)

        x = F.leaky_relu(self.deconv0(x), negative_slope=0.2)
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.2)

        img = torch.sigmoid(self.conv(x))

        return img