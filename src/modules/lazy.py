import torch


class LazyLayer(torch.nn.Module):
    """
    Currently a single elementwise multiplication with one laziness parameter per
    channel. This is run through a softmax so that this is a real laziness parameter.
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
