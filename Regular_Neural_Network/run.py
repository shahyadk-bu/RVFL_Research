import os
import csv
import torch

from model import NeuralNetwork
from trainer import train_model
from utils import load_mnist_tensors


def save_history_csv(history, filepath):
    keys = list(history.keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(len(history[keys[0]])):
            writer.writerow([history[k][i] for k in keys])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train, y_train_onehot, X_test, y_test_onehot, y_train_labels, y_test_labels = load_mnist_tensors(
    device=device,
    dtype=torch.float32
)

    input_size = X_train.shape[1]
    output_size = 10

    experiments = [
        {"width": 50, "target_acc": 0.90},
        {"width": 150, "target_acc": 0.94},
        {"width": 250, "target_acc": 0.96},
    ]

    save_dir = "Regular_Neural_Network/standard_nn_target_results_lr3e-2"
    os.makedirs(save_dir, exist_ok=True)

    summary_path = os.path.join(save_dir, "summary.csv")

    summary_rows = []

    for exp in experiments:
        width = exp["width"]
        target_acc = exp["target_acc"]

        print("\n" + "=" * 60)
        print(f"Running width={width}, target_acc={target_acc}")
        print("=" * 60)

        model = NeuralNetwork(
            layer_sizes=[input_size, width, output_size],
            activation=torch.relu
        ).to(device)

        history, summary = train_model(
            model=model,
            X_train=X_train,
            y_train_labels=y_train_labels,
            epochs=150,
            lr=3e-2,
            target_acc=target_acc
        )

        history_file = os.path.join(
            save_dir,
            f"width{width}_target{int(target_acc * 100)}_history.csv"
        )

        save_history_csv(history, history_file)

        summary_rows.append({
            "width": width,
            "target_acc": target_acc,
            "hit_target_epoch": summary["hit_target_epoch"],
            "hit_target_time": summary["hit_target_time"],
            "final_acc": summary["final_acc"],
            "final_loss": summary["final_loss"],
            "total_training_time": summary["total_training_time"],
            "history_file": history_file,
        })

    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "width",
            "target_acc",
            "hit_target_epoch",
            "hit_target_time",
            "final_acc",
            "final_loss",
            "total_training_time",
            "history_file",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nFinished all experiments.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()