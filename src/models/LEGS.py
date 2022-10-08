import torch
from modules import Scatter
from torch.nn import Linear

# Author: Alex Tong
# Reference: Data-Driven Learning of Geometric Scattering Networks, IEEE Machine Learning for Signal Processing Workshop 2021


class TSNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, edge_in_channels=None, trainable_laziness=False, **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_laziness = trainable_laziness
        self.scatter = Scatter(
            in_channels, trainable_laziness=trainable_laziness)
        self.lin1 = Linear(self.scatter.out_shape(), out_channels)
        self.act = torch.nn.LeakyReLU()

    def forward(self, data):
        x, sc = self.scatter(data)
        x = self.act(x)
        x = self.lin1(x)
        return x, sc
