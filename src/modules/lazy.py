import torch


class LazinessLayer(torch.nn.Module):
    """
    Currently a single elementwise multiplication with one laziness parameter per
    channel. This is run through a sigmoid so that this is a real laziness parameter.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.laziness_logit = torch.nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor, propagated: torch.Tensor) -> torch.Tensor:
        laziness = torch.nn.functional.sigmoid(self.laziness_logit)
        laziness = torch.unsqueeze(laziness, dim=1)
        return laziness * x + (1 - laziness) * propagated

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.weights)
