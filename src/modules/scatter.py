import torch

from .diffuse import Diffuse


class Scatter(torch.nn.Module):
    """
    Learnable Geometric Scattering Module.
    https://arxiv.org/pdf/2208.07458.pdf

    Quoting the paper:
        1. The geometric scattering transform consists of a cascade of graph filters
        constructed from a left stochastic diffusion matrix P := 1/2 (I + W D^-1),
        which corresponds to transition probabilities of a lazy random walk Markov process.
        2. Laziness := the probability of staying at the node instead of moving to a neighbor.
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

        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=torch.float, requires_grad=True))

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        s0 = x[:, :, None]
        avgs = [s0]
        for i in range(16):
            avgs.append(self.diffusion_layer1(avgs[-1], edge_index))
        for j in range(len(avgs)):
            # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
            avgs[j] = avgs[j][None, :, :, :]

        # Combine the diffusion levels into a single tensor.
        diffusion_levels = torch.cat(avgs)

        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter1 = avgs[1] - avgs[2]
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16]
        subtracted = torch.matmul(
            self.wavelet_constructor, diffusion_levels.view(17, -1))
        # reshape into given input shape
        subtracted = subtracted.view(4, x.shape[0], x.shape[1])
        s1 = torch.abs(
            torch.transpose(torch.transpose(subtracted, 0, 1), 1, 2))  # transpose the dimensions to match previous

        # perform a second wave of diffusing, on the recently diffused.
        avgs = [s1]
        for i in range(16):  # diffuse over diffusions
            avgs.append(self.diffusion_layer2(avgs[-1], edge_index))
        # add an extra dimension to each diffusion level for concatenation
        for i in range(len(avgs)):
            avgs[i] = avgs[i][None, :, :, :]
        diffusion_levels2 = torch.cat(avgs)

        # Having now generated the diffusion levels, we can cmobine them as before
        subtracted2 = torch.matmul(
            self.wavelet_constructor, diffusion_levels2.view(17, -1))
        # reshape into given input shape
        subtracted2 = subtracted2.view(
            4, s1.shape[0], s1.shape[1], s1.shape[2])
        subtracted2 = torch.transpose(subtracted2, 0, 1)
        subtracted2 = torch.abs(subtracted2.reshape(-1, self.in_channels, 4))
        s2_swapped = torch.reshape(torch.transpose(
            subtracted2, 1, 2), (-1, 16, self.in_channels))
        s2 = s2_swapped[:, feng_filters()]

        x = torch.cat([s0, s1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, s2], dim=1)

        #x = scatter_mean(x, batch, dim=0)
        if hasattr(data, 'batch'):
            x = scatter_moments(x, data.batch, 4)
            # x = scatter_moments(x, torch.zeros(data.x.shape[0], dtype=torch.int32), 4)
        else:
            x = scatter_moments(x, torch.zeros(
                data.x.shape[0], dtype=torch.int32), 4)
            # print('x returned shape', x.shape)

        return x, self.wavelet_constructor

    def out_shape(self):

        # x * 4 moments * in
        return 11 * 4 * self.in_channels


def scatter_moments(graph: torch.Tensor, batch_indices: torch.Tensor,
                    moments_returned: int = 4, inf_val: int = 1e15) -> torch.Tensor:
    """
    Compute specified statistical coefficients for each feature of each graph passed.
        `graph`: The feature tensors of disjoint subgraphs within a single graph.
            [N, in_channels] where N := number of nodes
        `batch_indices`: [B].
        `moments_returned`: Specifies the number of statistical measurements to compute.
            If 1, only the mean is returned. If 2, the mean and variance.
            If 3, the mean, variance, and skew. If 4, the mean, variance, skew, and kurtosis.
        `inf_val`: A value bigger than this shall be treated as infinity.
    """

    # Step 1: Aggregate the features of each mini-batch graph into its own tensor.
    graph_features = [torch.zeros(0).to(graph.device)
                      for _ in range(torch.max(batch_indices) + 1)]

    for i, node_features in enumerate(graph):
        # Sort the graph features by graph, according to batch_indices.
        # For each graph, create a tensor whose first row is the first element of each feature, etc.
        # print("node features are", node_features)

        if (len(graph_features[batch_indices[i]]) == 0):
            # If this is the first feature added to this graph, fill it in with the features.
            # .view(-1,1,1) changes [1,2,3] to [[1],[2],[3]], so that we can add each column to the respective row.
            graph_features[batch_indices[i]] = node_features.view(-1, 1, 1)
        else:
            graph_features[batch_indices[i]] = torch.cat(
                (graph_features[batch_indices[i]], node_features.view(-1, 1, 1)), dim=1)  # concatenates along columns

    # Instatiate the correct set of moments to return.
    assert moments_returned in [1, 2, 3, 4], \
        "`scatter_moments`: only supports `moments_returned` of the following values: 1, 2, 3, 4."
    moments_keys = ['mean', 'variance', 'skew', 'kurtosis']
    moments_keys = moments_keys[:moments_returned]
    statistical_moments = {}
    for key in moments_keys:
        statistical_moments[key] = torch.zeros(0).to(graph)

    for data in graph_features:

        data = data.squeeze()

        mean = torch.mean(data, dim=1, keepdim=True)

        if moments_returned >= 1:
            statistical_moments['mean'] = torch.cat(
                (statistical_moments['mean'], mean.T), dim=0
            )

        # produce matrix whose every row is data row - mean of data row
        std = data - mean

        # variance: difference of u and u mean, squared element wise, summed and divided by n-1
        variance = torch.mean(std**2, axis=1)
        if moments_returned >= 2:
            statistical_moments['variance'] = torch.cat(
                (statistical_moments['variance'], variance[None, ...]), dim=0
            )

        # skew: 3rd moment divided by cubed standard deviation (sd = sqrt variance), with correction for division by zero (inf -> 0)
        skew = variance = torch.mean(std**3, axis=1)
        # Multivalued tensor division by zero produces inf.
        skew[skew > inf_val] = 0
        # Single valued division by 0 produces nan.
        skew[skew != skew] = 0
        if moments_returned >= 3:
            statistical_moments['skew'] = torch.cat(
                (statistical_moments['skew'], skew[None, ...]), dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition to subtract 3 (default in scipy)
        kurtosis = torch.mean(std**4, axis=1) - 3
        kurtosis[kurtosis > inf_val] = -3
        kurtosis[kurtosis != kurtosis] = -3
        if moments_returned >= 4:
            statistical_moments['kurtosis'] = torch.cat(
                (statistical_moments['kurtosis'], kurtosis[None, ...]), dim=0
            )

    # Concatenate into one tensor.
    statistical_moments = torch.cat(
        [statistical_moments[key] for key in moments_keys], axis=1)

    return statistical_moments


def feng_filters():
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4*i+j)

    return results
