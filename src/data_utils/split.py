import numpy as np
import torch


def split_dataset(dataset, splits=(0.8, 0.1, 0.1), seed=0):
    """ Splits data into non-overlapping datasets of given proportions.
    """

    splits = np.array(splits)
    splits = splits / np.sum(splits)
    n = len(dataset)
    torch.manual_seed(seed)
    val_size = int(splits[1] * n)
    test_size = int(splits[2] * n)
    train_size = n - val_size - test_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    return train_set, val_set, test_set
