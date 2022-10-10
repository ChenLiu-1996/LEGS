from typing import Optional, Tuple

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from .lazy import LazinessLayer


class Diffuse(MessagePassing):
    """
    Low pass walk with optional weights.
    Init @params
        `in_channels`:        number of input channels
        `out_channels`:       number of output channels
        `trainable_laziness`: whether the "laziness" (probability of not moving to neighbor) is trainable.
        `fixed_weights`:      whether or not to linearly transform the node feature matrix.
    Forward @params
        `x`:                  input graph  [N, in_channels] where N := number of nodes
        `edge_index`:         edge indices [2, E] where E := number of edges
        `edge_weight`:        edge weights [E] where E := number of edges
    """

    def __init__(self, in_channels: int, out_channels: int,
                 trainable_laziness: bool = False, laziness: Optional[float] = 0.5, fixed_weights: bool = True) -> None:

        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.laziness_layer = LazinessLayer(in_channels)
        else:
            self.laziness = laziness
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):

        # Step 2: Linearly transform node feature matrix.
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(
            edge_index=edge_index, edge_weight=edge_weight, num_nodes=x.size(self.node_dim), dtype=x.dtype)

        # Step 4-6: Message-passing.
        propogated = self.propagate(
            edge_index, edge_weight=edge_weight, size=None, x=x,
        )
        if not self.trainable_laziness:
            return self.laziness * x + (1 - self.laziness) * propogated
        else:
            return self.laziness_layer(x, propogated)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:

        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j

    def message_and_aggregate(self, adj_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return torch.matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:

        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out


def gcn_norm(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = False, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize the edge weight by edge degree?
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index=edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    node_from, node_to = edge_index[0, :], edge_index[1, :]

    # `scatter_add`: Sums all values from the `src` tensor into `out`
    # at the indices specified in the index tensor along a given axis `dim`.
    deg = scatter_add(src=edge_weight, index=node_to, dim=0,
                      out=None, dim_size=num_nodes)

    # Don't use tensor.pow_(-1). It will modify tensor in-place!!!
    # TODO: Why is it called 'deg_inv_SQRT'??
    deg_inv_sqrt = torch.pow(deg, -1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    return edge_index, deg_inv_sqrt[node_from] * edge_weight
