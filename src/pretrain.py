import os

import torch
import torch.utils
from data_utils import ZINCDataset, split_dataset
from models import TSNet
from torch_geometric.loader import DataLoader
from tqdm import trange
from utils import EarlyStopping, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model_dir, out_file):

    dataset = ZINCDataset(model_dir, prop_stat_dict=None, include_ki=False)
    # print(dataset)
    train_ds, val_ds, test_ds = split_dataset(dataset)
    # reduce the num_workers to get rid of RuntimeError?
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=0)

    # print("done data loading")
    model = TSNet(
        dataset.num_node_features,
        dataset.num_classes,
        trainable_laziness=False
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(mode='max', patience=5, percentage=True)

    results_compiled = []
    early_stopper = EarlyStopping(mode='min', patience=5, percentage=False)

    model.train()

    for epoch in trange(1, 80 + 1):

        for data in train_loader:

            optimizer.zero_grad()
            data = data.to(device)
            out, sc = model(data)

            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()

        results = evaluate(model, loss_fn, train_ds, test_ds, val_ds, device)
        #print('Epoch:', epoch, results['train_acc'], results['test_acc'])
        results_compiled.append(results['test_acc'])

        if early_stopper.step(results['val_acc']):
            print("Early stopping criterion met. Ending training.")
            # if the validation accuracy decreases for eight consecutive epochs, break.
            break

    model.eval()

    results = evaluate(model, loss_fn, train_ds, test_ds, val_ds, device)
    print("Results compiled:", results_compiled)

    # print('saving scatter model')
    # torch.save(model.scatter.state_dict(), str(out_file) + f"IMiD_weights.npy")


if __name__ == '__main__':
    import subprocess
    current_dir = subprocess.check_output(
        "pwd", shell=True).decode("utf-8").split('\n')[0]
    os.makedirs(current_dir, exist_ok=True)
    train_model(model_dir='/'.join(current_dir.split('/')[:-1]) + '/data/IMiD_smiles.npy',
                out_file=current_dir + '/trained_models/')
