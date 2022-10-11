import torch
from modules import Scatter
from torch.nn import Linear


# Reference: Data-Driven Learning of Geometric Scattering Networks, IEEE Machine Learning for Signal Processing Workshop 2021


class LEGSNet(torch.nn.Module):
    """
    Learnable Geometric Scattering Network.
    `in_channels`:        number of input channels.
    `out_channels`:       number of output channels.
    `trainable_laziness`: whether the "laziness" (probability of not moving to neighbor) is trainable.
    """

    def __init__(self, in_channels: int, out_channels: int, trainable_laziness: bool = False) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trainable_laziness = trainable_laziness
        self.scatter = Scatter(
            in_channels, trainable_laziness=trainable_laziness)
        self.act = torch.nn.LeakyReLU()
        self.fc = Linear(self.scatter.out_shape(), out_channels)

    def forward(self, data):
        x, sc = self.scatter(data)
        x = self.act(x)
        x = self.fc(x)
        return x, sc
