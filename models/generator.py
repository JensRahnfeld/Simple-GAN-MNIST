import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim_latent=100):
        super(Generator, self).__init__()
        
        self.dim_latent = dim_latent
        self.layers = [512, 256, 128]
        
        self.fc = nn.Linear(dim_latent, self.layers[0] * 3 * 3)
        self.deconv0 = nn.ConvTranspose2d(self.layers[0], self.layers[1],
                                            kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(self.layers[1], self.layers[2],
                                            kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.layers[2], 1, kernel_size=6, stride=2)

        self.norm0 = nn.InstanceNorm2d(self.layers[0])
        self.norm1 = nn.InstanceNorm2d(self.layers[1])
        self.norm2 = nn.InstanceNorm2d(self.layers[2])

    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.fc(x))
        x = x.view(batch_size, self.layers[0], 3, 3)
        
        x = self.norm0(x)
        x = F.leaky_relu(self.deconv0(x), negative_slope=0.2)
        
        x = self.norm1(x)
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.2)

        x = self.norm2(x)
        img = torch.sigmoid(self.deconv2(x))

        return img