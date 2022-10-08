from torch_geometric.loader import DataLoader


def accuracy(model, dataset, loss_fn, name, device):

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_loss = 0

    for data in loader:
        data = data.to(device)
        pred, sc = model(data)
        total_loss += loss_fn(pred, data.y)

    acc = total_loss / len(dataset)

    return acc, pred
