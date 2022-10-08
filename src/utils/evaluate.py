from .metrics import accuracy


def evaluate(model, loss_fn, train_ds, test_ds, val_ds, device):

    train_acc, train_pred = accuracy(model, train_ds, loss_fn, "Train", device)
    val_acc, val_pred = accuracy(model, val_ds, loss_fn, "Validation", device)
    test_acc, test_pred = accuracy(model, test_ds, loss_fn, "Test", device)

    results = {
        "train_acc": train_acc,
        "train_pred": train_pred,
        "test_acc": test_acc,
        "test_pred": test_pred,
        "val_acc": val_acc,
        "val_pred": val_pred,
        "state_dict": model.state_dict(),
    }

    return results
