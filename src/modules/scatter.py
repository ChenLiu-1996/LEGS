from typing import Union

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from .aggregate import Aggregate
from .diffuse import Diffuse


class Scatter(torch.nn.Module):
    """
    "The Scattering submodule" in https://arxiv.org/pdf/2208.07458.pdf.

    Quoting the paper:
        1. The geometric scattering transform consists of a cascade of graph filters
        constructed from a left stochastic diffusion matrix P := 1/2 (I + W D^-1),
        which corresponds to transition probabilities of a lazy random walk Markov process.
        2. Laziness := the probability of staying at the node instead of moving to a neighbor.

    Init @params
        `in_channels`:        number of input channels
        `trainable_laziness`: whether the "laziness" (probability of not moving to neighbor) is trainable.
    Forward @params
        `graph_data`:         torch_geometric.data.Data or torch_geometric.data.batch.Batch
                              with fields graph_data.x (node features) and graph_data.edge_index

    Math:
    `P`:   diffusion matrices.
    `Psi`: graph wavelet matrices.
    """

    def __init__(self, in_channels: int, trainable_laziness: bool = False) -> None:
        super(Scatter, self).__init__()

        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness

        self.diffusion_layer1 = Diffuse(
            in_channels=in_channels,
            out_channels=in_channels,
            trainable_laziness=trainable_laziness)

        self.diffusion_layer2 = Diffuse(
            in_channels=4*in_channels,
            out_channels=4*in_channels,
            trainable_laziness=trainable_laziness
        )

        # Weightings for the 0th to 2^4th diffusion wavelets.
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=torch.float, requires_grad=True))

        self.aggregation_submodule = Aggregate(
            aggregation_method='statistical_moments')

    def forward(self, graph_data: Union[Data, Batch]):
        # TODO: Still need to go through the `forward` function
        # and cross-compare with the paper...

        x, edge_index = graph_data.x, graph_data.edge_index

        # 0th scattering moments.
        S0 = x[:, :, None]
        diffusion_matrices = [S0]
        for _ in range(2**4):
            diffusion_matrices.append(self.diffusion_layer1(
                diffusion_matrices[-1], edge_index))
        # Combine the diffusion levels into a single tensor.
        # Diffusion level 1.
        diffusion_levels = torch.stack(diffusion_matrices)

        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter1 = avgs[1] - avgs[2]
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16]
        subtracted = torch.matmul(
            self.wavelet_constructor, diffusion_levels.view(self.wavelet_constructor.shape[-1], -1))
        # reshape into given input shape
        subtracted = subtracted.view(4, x.shape[0], x.shape[1])

        # 1st scattering moments.
        S1 = torch.abs(
            torch.transpose(torch.transpose(subtracted, 0, 1), 1, 2))  # transpose the dimensions to match previous

        # Perform a second wave of diffusing, on the recently diffused.
        diffusion_matrices = [S1]
        for i in range(2**4):  # diffuse over diffusions
            diffusion_matrices.append(self.diffusion_layer2(
                diffusion_matrices[-1], edge_index))
        # Diffusion level 2.
        diffusion_levels = torch.stack(diffusion_matrices)

        # Having now generated the diffusion levels, we can cmobine them as before
        subtracted = torch.matmul(self.wavelet_constructor, diffusion_levels.view(
            self.wavelet_constructor.shape[-1], -1))
        # reshape into given input shape
        subtracted = subtracted.view(4, S1.shape[0], S1.shape[1], S1.shape[2])
        subtracted = torch.transpose(subtracted, 0, 1)
        subtracted = torch.abs(subtracted.reshape(-1, self.in_channels, 4))
        S2_swapped = torch.reshape(torch.transpose(
            subtracted, 1, 2), (-1, 16, self.in_channels))

        # 2nd scattering moments.
        S2 = S2_swapped[:, feng_filters()]

        x = torch.cat([S0, S1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, S2], dim=1)

        #x = scatter_mean(x, batch, dim=0)
        if hasattr(graph_data, 'batch'):
            x = self.aggregation_submodule(
                graph=x, batch_indices=graph_data.batch, moments_returned=4)
        else:
            x = self.aggregation_submodule(graph=x, batch_indices=torch.zeros(
                graph_data.x.shape[0], dtype=torch.int32), moments_returned=4)

        return x, self.wavelet_constructor

    def out_shape(self):
        # x * 4 moments * in
        return 11 * 4 * self.in_channels


def feng_filters():
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4*i+j)

    return results
