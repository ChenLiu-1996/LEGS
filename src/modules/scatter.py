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

    Technical details:
        The Geometric Scattering Process is formulated as several (2 in this implementation)
        [diffusion, scattering] blocks followed by a final aggregation block.
        In each [diffusion, scattering] block, there are (1+2^J) diffusion steps and
        J scattering filters involved. J represents the "order" of diffusion.
        In this implementation, J is 4.

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

        ''' Pre-Diffusion/Scattering round 1 '''
        # Scattering round 0 outputs := input.
        S0 = x[:, :, None]

        ''' Diffusion round 1 '''
        diffusion_outputs = [S0]
        # Let the scattering outcome go through 2^4 diffusion steps.
        for _ in range(2**4):
            diffusion_outputs.append(self.diffusion_layer1(
                diffusion_outputs[-1], edge_index))
        # Stack the diffusion outputs into a single tensor.
        # This represents [P^0, P^1, ..., P^2^J]
        diffusion_outputs = torch.stack(diffusion_outputs)

        ''' Scattering round 1 '''
        # This simulates the below subtraction:
        #   psi_j = P^2^(j-1) - P^2^j for 0<=j<=J=4
        # Specifically, the implementation directly takes the signal into the process:
        #   psi_j x = (P^2^(j-1) - P^2^j) x
        scattering_outputs = torch.matmul(
            self.wavelet_constructor, diffusion_outputs.view(self.wavelet_constructor.shape[-1], -1))
        scattering_outputs = scattering_outputs.view(4, x.shape[0], x.shape[1])

        ''' Pre-Diffusion/Scattering round 2 '''
        # Scattering round 1 outputs.
        S1 = torch.abs(
            torch.transpose(torch.transpose(scattering_outputs, 0, 1), 1, 2))  # transpose the dimensions to match previous

        ''' Diffusion round 2 '''
        diffusion_outputs = [S1]
        for _ in range(2**4):
            diffusion_outputs.append(self.diffusion_layer2(
                diffusion_outputs[-1], edge_index))
        diffusion_outputs = torch.stack(diffusion_outputs)

        ''' Scattering round 2 '''
        scattering_outputs = torch.matmul(self.wavelet_constructor, diffusion_outputs.view(
            self.wavelet_constructor.shape[-1], -1))
        scattering_outputs = scattering_outputs.view(
            4, S1.shape[0], S1.shape[1], S1.shape[2])
        scattering_outputs = torch.transpose(scattering_outputs, 0, 1)
        scattering_outputs = torch.abs(
            scattering_outputs.reshape(-1, self.in_channels, 4))
        scattering_outputs = torch.reshape(torch.transpose(
            scattering_outputs, 1, 2), (-1, 16, self.in_channels))

        # Scattering round 2 outputs.
        S2 = scattering_outputs[:, feng_filters()]

        ''' Aggregation after all [diffusion, scattering] blocks '''
        x = torch.cat([S0, S1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, S2], dim=1)

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
