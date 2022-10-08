import argparse
import os
from typing import Tuple

import torch
import torch.utils
import yaml
from data_utils import ZINCDataset, split_dataset
from models import TSNet
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import trange
from utils import AttributeHashmap, EarlyStopping, compute_metrics, log


def parse_config(config: AttributeHashmap) -> AttributeHashmap:
    config.log_dir = config.log_folder + \
        os.path.basename(config.config_file_name).rstrip('.yaml') + '_log.txt'

    # Resolve type issues.
    config.learning_rate = float(config.learning_rate)

    # Initialize log file.
    log_str = 'Config: \n\n'
    for key in config.keys():
        log_str += '%s: %s\n' % (key, config[key])
    log_str += '\n\nTraining History: \n\n'
    log(log_str, filepath=config.log_dir, to_console=True)

    return config


def prepare_dataset(config: AttributeHashmap) -> Tuple[Dataset, DataLoader, DataLoader, DataLoader]:
    dataset = ZINCDataset(config.dataset_dir,
                          prop_stat_dict=None, include_ki=False)

    split_ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    train_ds, val_ds, test_ds = split_dataset(
        dataset, splits=split_ratios, random_seed=config.random_seed)

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

    model = TSNet(
        in_channels=dataset.num_node_features,
        out_channels=dataset.num_classes,
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
            val_loss, val_MAE, val_MSE = compute_metrics(metrics=['loss', 'MAE', 'MSE'], model=model, loader=val_loader,
                                                         loss_fn=loss_fn, device=config.device)

        log('Epoch (%d/%d), val loss: %.6f, MAE by property: %s, MSE by property: %s' %
            (epoch_idx, config.max_epochs, val_loss, val_MAE, val_MSE), filepath=config.log_dir, to_console=False)

        if early_stopper.step(val_loss):
            print("Early stopping criterion met. Ending training.")
            # if the validation accuracy decreases for eight consecutive epochs, break.
            break

    with torch.no_grad():
        test_loss, test_MAE, test_MSE = compute_metrics(metrics=['loss', 'MAE', 'MSE'], model=model, loader=test_loader,
                                                        loss_fn=loss_fn, device=config.device)
    log('Final model, test loss: %.6f, MAE by property: %s, MSE by property: %s' %
        (test_loss, test_MAE, test_MSE), filepath=config.log_dir, to_console=False)

    # print('saving scatter model')
    # torch.save(model.scatter.state_dict(), str(save_folder) + f"IMiD_weights.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entry point to run LEGS.')
    parser.add_argument(
        '--pretrain', help='Flag for running pretraining.', action='store_true')
    parser.add_argument(
        '--config', help='Path to config yaml file.', required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)

    # Load the config yaml file and store it as a Attribute Hashmap.
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_config(config)

    # Automatically detect and assign device.
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.random_seed)
    if args.pretrain:
        pretrain_model(config)
