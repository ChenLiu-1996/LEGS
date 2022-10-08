import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


def accuracy(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:

    total_loss = 0

    for data in loader:
        data = data.to(device)
        pred, sc = model(data)
        total_loss += loss_fn(pred, data.y).item()

    acc = total_loss / len(loader)

    return acc
