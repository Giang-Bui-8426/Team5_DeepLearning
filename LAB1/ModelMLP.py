from __future__ import annotations

from collections.abc import Sequence
from platform import architecture
from turtle import width

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channel,activation,normalization_layer, dropout):
        super().__init__()
        self.block = nn.Sequential(
            normalization_layer(channel),
            nn.Linear(channel, channel*2),
            activation,
            nn.Linear(channel*2, channel),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class MLP(nn.Module):
    def __init__(self,input=28*28,hidden= (256, 128),
        activation= "LeakyRelu",normalization= "none",
        dropout= 0.0,residual_blocks= 0,num_classes= 10):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        def normalization_layer(channel):
            if normalization == "batchnorm":
                return nn.BatchNorm1d(channel)
            elif normalization == "layernorm":
                return nn.LayerNorm(channel)
            else:
                return nn.Identity()
        activation = nn.LeakyReLU() 
        if residual_blocks == 0:
            layers = [] # backbone of the MLP
            for channel in hidden:
                layers.append(nn.Linear(input, channel))
                layers.append(normalization_layer(channel))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
                input = channel
            layers.append(nn.Linear(input, num_classes)) # layer output
            
            self.network = nn.Sequential(*layers)
        elif residual_blocks > 0:
            channel = hidden[0]
            self.fc1 = nn.Sequential(
                nn.Linear(input, channel),
                normalization_layer(channel),
                activation,
            )
            self.fc2 = nn.Sequential(
                nn.Linear(channel, channel*2),
                normalization_layer(channel*2),
                activation,
                nn.Dropout(dropout)
            )
            self.residual_blocks = nn.Sequential(
                *[ResidualBlock(channel*2, activation, normalization_layer, dropout) for _ in range(residual_blocks)],
                normalization_layer(channel*2),
            )
            self.fc4 = nn.Sequential(
                nn.Linear(channel*2, channel),
                normalization_layer(channel),
                activation,
            )
            self.fc5 = nn.Linear(channel, num_classes)
            self.network = nn.Sequential(
                self.fc1,
                self.fc2,
                self.residual_blocks,
                self.fc4,
                self.fc5,
            )
        else:
            raise ValueError(f"Error: residual_blocks is not a valid number")

    def forward(self, x):
        # [B, 1, 28, 28] -> [B, 784]
        x = self.flatten(x)
        # [B, 784] -> [B, 10]
        return self.network(x)
