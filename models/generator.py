import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_layers import _ConvTranspose2d


class Generator(nn.Module):
    def __init__(self, dim_latent=100):
        super(Generator, self).__init__()

        self.dim_latent = dim_latent
        self.layers = [512, 256, 128]
        
        self.fc = nn.Linear(dim_latent, self.layers[0] * 3 * 3)
        
        self.deconv0 = _ConvTranspose2d(self.layers[0], self.layers[1],
                                        kernel_size=4, stride=2, padding=1,
                                        norm="batch", activation="leaky_relu")
        self.deconv1 = _ConvTranspose2d(self.layers[1], self.layers[2],
                                        kernel_size=4, stride=2, padding=1,
                                        norm="batch", activation="leaky_relu")
        self.deconv2 = _ConvTranspose2d(self.layers[2], 1, kernel_size=6, stride=2,
                                        norm="none", activation="sigmoid")

    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.fc(x))
        x = x.view(batch_size, self.layers[0], 3, 3)
        
        x = self.deconv0(x)
        x = self.deconv1(x)
        img = self.deconv2(x)

        return img