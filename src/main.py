import argparse
from typing import Tuple

import torch
import torch.utils
import yaml
from data_utils import ZINCDataset, split_dataset
from models import TSNet
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import trange
from utils import AttributeHashmap, EarlyStopping, accuracy


def parse_settings(config: AttributeHashmap) -> AttributeHashmap:
    # type issues
    config.learning_rate = float(config.learning_rate)
    return config


def prepare_dataset(config: AttributeHashmap) -> Tuple[Dataset, DataLoader, DataLoader, DataLoader]:
    dataset = ZINCDataset(config.dataset_dir,
                          prop_stat_dict=None, include_ki=False)

    split_ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    train_ds, val_ds, test_ds = split_dataset(
        dataset, splits=split_ratios, seed=config.random_seed)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds),
                            shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds),
                             shuffle=False, num_workers=config.num_workers)
    return dataset, train_loader, val_loader, test_loader


def pretrain_model(config: AttributeHashmap) -> None:
    """
    Pretrain the LEGS model and evaluate its accuracy.
    """

    dataset, train_loader, val_loader, test_loader = prepare_dataset(config)

    # print("done data loading")
    model = TSNet(
        dataset.num_node_features,
        dataset.num_classes,
        trainable_laziness=False
    )

    model = model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(
        mode='min', patience=config.patience, percentage=False)

    model.train()

    for epoch_idx in trange(1, config.max_epochs + 1):
        for train_data in train_loader:
            train_data = train_data.to(config.device)

            optimizer.zero_grad()

            out, sc = model(train_data)
            loss = loss_fn(out, train_data.y)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc, _ = accuracy(model, val_loader, loss_fn, config.device)

        print('Epoch (%s/%s), val acc: %s' %
              (epoch_idx, config.max_epochs, val_acc))

        if early_stopper.step(val_acc):
            print("Early stopping criterion met. Ending training.")
            # if the validation accuracy decreases for eight consecutive epochs, break.
            break

    with torch.no_grad():
        test_acc, _ = accuracy(model, test_loader, loss_fn, config.device)
    print('Final test acc: %s' % test_acc)

    # print('saving scatter model')
    # torch.save(model.scatter.state_dict(), str(save_dir) + f"IMiD_weights.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entry point to run LEGS.')
    parser.add_argument(
        '--pretrain', help='Flag for running pretraining.', action='store_true')
    parser.add_argument(
        '--config', help='Path to config yaml file.', required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = parse_settings(config)

    config.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.random_seed)
    if config.pretrain:
        pretrain_model(config)
