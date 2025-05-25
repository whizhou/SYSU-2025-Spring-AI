import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # nn.GroupNorm(n_groups, out_channels),
            # nn.Mish(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.ModuleList([
            Conv2dBlock(in_channels, out_channels, kernel_size=3),
            Conv2dBlock(out_channels, out_channels, kernel_size=3),
        ])
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()


    def forward(self, x):
        out = self.block[0](x)
        out = self.block[1](out)
        out = out + self.residual_conv(x)
        return out

class ChineseHerbModel(nn.Module):
    def __init__(self, base_channels=32, output_dim=5, down_dims = [64, 128, 256, 512]):
        super(ChineseHerbModel, self).__init__()
        # down_dims = [64, 128, 256, 512, 1024]
        all_dims = [base_channels] + down_dims
        self.conv_in = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1)
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(ResidualBlock2D(all_dims[i], all_dims[i+1]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_layers = nn.Sequential(*layers)

        width = 4
        self.global_avg_pool = nn.AdaptiveAvgPool2d(width)  # 全局平均池化
        num_features = all_dims[-1] * (width ** 2)
        self.fc = nn.Linear(num_features, output_dim)

        self.down_dims = down_dims
        self.base_channels = base_channels
        self.output_dim = output_dim
        self.num_features = num_features
        self.all_dims = all_dims

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # 展平为 [batch_size, num_features]
        x = self.fc(x)
        return x
