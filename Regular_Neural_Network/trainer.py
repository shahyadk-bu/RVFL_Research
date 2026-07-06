import time
import torch
import torch.nn as nn

def accuracy(model, X, y_labels):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_labels).float().mean().item()

    return acc

def train_model(model, X_train, y_train_labels, epochs=50, lr=1e-3, target_acc=None):
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "loss": [],
        "acc": [],
        "epoch_time": [],
        "total_time": [],
    }

    hit_target_epoch = None
    hit_target_time = None

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad()

        logits = model(X_train)
        loss = lossFunc(logits, y_train_labels)

        loss.backward()
        optimizer.step()

        train_acc = accuracy(model, X_train, y_train_labels)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start

        history["loss"].append(loss.item())
        history["acc"].append(train_acc)
        history["epoch_time"].append(epoch_time)
        history["total_time"].append(total_time)

        if target_acc is not None and hit_target_epoch is None:
            if train_acc >= target_acc:
                hit_target_epoch = epoch + 1
                hit_target_time = total_time

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={loss.item():.6f} | "
            f"acc={train_acc:.4f} | "
            f"epoch_time={epoch_time:.2f}s | "
            f"total_time={total_time:.2f}s"
        )

    summary = {
        "total_training_time": time.time() - training_start,
        "final_loss": history["loss"][-1],
        "final_acc": history["acc"][-1],
        "target_acc": target_acc,
        "hit_target_epoch": hit_target_epoch,
        "hit_target_time": hit_target_time,
    }

    return history, summary