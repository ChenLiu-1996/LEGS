from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


def compute_metrics(metrics: List[str] = ['MAE', 'MSE', 'loss'],
                    model: nn.Module = None, loader: DataLoader = None, loss_fn: nn.Module = None,
                    device: torch.device = None) -> Union[float, Tuple[float]]:
    """
    This function makes quite a few assumptions.
    1. `model` produces 2 outputs, the first of which is
        the predicted properties `pred` with dimension [B, K]
            B: batch dimension
            K: number of properties to predict
    2. Ground truth is represented in `graph_data.y`, with the same dimension as `pred`.
    3. Since it's a regression/prediction task, it makes sense to use the following metrics:
        - MAE: mean absolute error (calculated per-property)
        - MSE: mean squared error (calculated per-property)
    """

    assert type(metrics) == type([])  # Make sure it's a List.

    mean_abs_err, mean_sqr_err, loss = None, None, None

    for graph_data in loader:
        graph_data = graph_data.to(device)
        pred, sc = model(graph_data)
        assert len(pred.shape) == 2
        assert pred.shape == graph_data.y.shape

        if 'loss' in metrics:
            curr_result = loss_fn(pred, graph_data.y).item()
            loss = curr_result if loss is None else loss + curr_result
        # For MAE and MSE.
        # Aggregating and taking the mean in the final step.
        # However, this still does not take into account the
        # potential error in case of unequal batch size.
        if 'MAE' in metrics:
            curr_result = np.array([
                F.l1_loss(
                    pred[:, i], graph_data.y[:, i], reduction='mean'
                ).cpu().detach().numpy() for i in range(pred.shape[1])])
            mean_abs_err = curr_result if mean_abs_err is None \
                else np.vstack((mean_abs_err, curr_result))
        if 'MSE' in metrics:
            curr_result = np.array([
                F.mse_loss(
                    pred[:, i], graph_data.y[:, i], reduction='mean'
                ).cpu().detach().numpy() for i in range(pred.shape[1])])
            mean_sqr_err = curr_result if mean_sqr_err is None \
                else np.vstack((mean_sqr_err, curr_result))

    loss = loss / len(loader)

    # len(mean_abs_err.shape) == 1 if dataset passed in 1 batch.
    assert len(mean_abs_err.shape) in [1, 2]
    assert mean_abs_err.shape == mean_sqr_err.shape

    if len(mean_abs_err.shape) == 2:
        mean_abs_err = np.mean(mean_abs_err, axis=0)
        mean_sqr_err = np.mean(mean_sqr_err, axis=0)

    return_items = []
    for item in metrics:
        if item == 'MAE':
            return_items.append(mean_abs_err)
        elif item == 'MSE':
            return_items.append(mean_sqr_err)
        elif item == 'loss':
            return_items.append(loss)
        else:
            raise NotImplementedError

    if len(return_items) > 1:
        return tuple(return_items)
    else:
        return return_items[0]
