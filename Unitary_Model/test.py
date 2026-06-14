import os
import csv
import torch
import matplotlib.pyplot as plt

from RVFL_Model import RVFL
from utils import load_mnist_tensors


def save_history_csv(history, filepath):
    keys = list(history.keys())
    num_rows = max(len(history[k]) for k in keys)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(num_rows):
            row = []
            for k in keys:
                if i < len(history[k]):
                    row.append(history[k][i])
                else:
                    row.append("")
            writer.writerow(row)


def plot_history(history, width, save_dir):
    plt.figure(figsize=(8, 5))

    if len(history["train_acc"]) > 0:
        plt.plot(history["train_acc"], label="Train Accuracy")
    if len(history["val_acc"]) > 0:
        plt.plot(history["val_acc"], label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"RVFL Unitary Training Accuracy (width={width})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, f"accuracy_width_{width}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def run_experiment(
    width,
    X_train,
    y_train,
    X_test,
    y_test,
    y_train_labels,
    y_test_labels,
    device,
    dtype,
    epochs=15,
    lr=1e-3,
):
    layersInfo = [
        {
            "layer_dim": width,
            "weight_dist": "normal",
            "weight_var": 1.0,
            "gamma_k": None,
            "bias_switch": False,
            "bias_dist": "normal",
            "bias_var": 1.0,
        }
    ]

    generalInfo = {
        "seed": 0,
        "device": device,
        "dtype": dtype,
    }

    model = RVFL(
        layersInfo=layersInfo,
        generalInfo=generalInfo,
        activation="relu",
        linkOption="direct",
        lamb=1e-3,
        scalings=[0.0],
    )

    history = model.train_Unitary(
        X_train=X_train,
        y_train=y_train,
        y_train_labels=y_train_labels,
        X_val=X_test,
        y_val=y_test,
        y_val_labels=y_test_labels,
        epochs=epochs,
        lr=lr,
        printUpdates=True,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, y_test_labels)

    theta_norm = model.unitaryParams[0].theta.norm().item()

    return model, history, test_loss, test_acc, theta_norm


def main():
    device = "cpu"
    dtype = torch.float64

    save_dir = "unitary_test_results"
    os.makedirs(save_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, y_train_labels, y_test_labels = load_mnist_tensors(
        device=device,
        dtype=dtype,
    )

    widths = [50]

    summary_rows = []

    for width in widths:
        print("\n" + "=" * 60)
        print(f"Running width = {width}")
        print("=" * 60)

        model, history, test_loss, test_acc, theta_norm = run_experiment(
            width=width,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_train_labels=y_train_labels,
            y_test_labels=y_test_labels,
            device=device,
            dtype=dtype,
            epochs=30,
            lr=1e-3,
        )

        csv_path = os.path.join(save_dir, f"history_width_{width}.csv")
        save_history_csv(history, csv_path)

        plot_history(history, width, save_dir)

        final_train_acc = history["train_acc"][-1] if len(history["train_acc"]) > 0 else None
        final_val_acc = history["val_acc"][-1] if len(history["val_acc"]) > 0 else None

        summary_rows.append({
            "width": width,
            "final_train_acc": final_train_acc,
            "final_test_acc": final_val_acc,
            "eval_test_acc": test_acc,
            "eval_test_loss": test_loss,
            "theta_norm": theta_norm,
        })

        print(f"Finished width={width}")
        print(f"Final recorded train acc: {final_train_acc}")
        print(f"Final recorded test acc:  {final_val_acc}")
        print(f"Evaluate() test acc:      {test_acc}")
        print(f"Evaluate() test loss:     {test_loss}")
        print(f"Theta norm:               {theta_norm}")

    summary_csv = os.path.join(save_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "width",
            "final_train_acc",
            "final_test_acc",
            "eval_test_acc",
            "eval_test_loss",
            "theta_norm",
        ])
        for row in summary_rows:
            writer.writerow([
                row["width"],
                row["final_train_acc"],
                row["final_test_acc"],
                row["eval_test_acc"],
                row["eval_test_loss"],
                row["theta_norm"],
            ])

    print("\nAll experiments complete.")
    print(f"Saved results in: {save_dir}")


if __name__ == "__main__":
    main()