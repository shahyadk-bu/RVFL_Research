import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Make imports work from the inner repo root and from the outer folder.
# Expected layout:
# outer/RVFL_Research/benchmarking/benchmark_job.py
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
INNER_REPO = THIS_FILE.parents[1]
OUTER_REPO = INNER_REPO.parent

sys.path.insert(0, str(INNER_REPO))
sys.path.insert(0, str(OUTER_REPO))


from Regular_Neural_Network.model import NeuralNetwork
from RVFL_Research.Unitary_Model.Model.RVFL_Model import RVFL
from RVFL_Research.Unitary_Model.Model.utils import (
    load_mnist_tensors,
    load_fashion_mnist_tensors,
    load_cifar10_tensors,
)


def sync_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history_csv(history, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    keys = list(history.keys())
    n = len(history[keys[0]])

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(n):
            writer.writerow([history[k][i] for k in keys])


def load_dataset(dataset_name, device, dtype):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        (
            X_train,
            y_train_onehot,
            X_test,
            y_test_onehot,
            y_train_labels,
            y_test_labels,
        ) = load_mnist_tensors(device=device, dtype=dtype)

        return {
            "X_train": X_train,
            "y_train": y_train_labels.long(),
            "X_test": X_test,
            "y_test": y_test_labels.long(),
            "input_dim": X_train.shape[1],
            "output_dim": 10,
            "task": "classification",
        }

    if dataset_name == "fashionmnist":
        data = load_fashion_mnist_tensors(
            device=device,
            dtype=dtype,
            normalize="standardize",
        )

        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"].long(),
            "X_test": data["X_test"],
            "y_test": data["y_test"].long(),
            "input_dim": data["input_dim"],
            "output_dim": data["output_dim"],
            "task": "classification",
        }

    if dataset_name == "cifar10":
        data = load_cifar10_tensors(
            device=device,
            dtype=dtype,
            normalize="standardize",
        )

        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"].long(),
            "X_test": data["X_test"],
            "y_test": data["y_test"].long(),
            "input_dim": data["input_dim"],
            "output_dim": data["output_dim"],
            "task": "classification",
        }

    raise ValueError(f"Unknown dataset: {dataset_name}")


def apply_debug_limits(data, train_limit=None, test_limit=None):
    """
    Optional smoke-test mode.

    The real benchmark uses the full dataset because the defaults are None.
    """
    if train_limit is not None:
        data["X_train"] = data["X_train"][:train_limit]
        data["y_train"] = data["y_train"][:train_limit]

    if test_limit is not None:
        data["X_test"] = data["X_test"][:test_limit]
        data["y_test"] = data["y_test"][:test_limit]

    return data

@torch.no_grad()
def classification_accuracy_from_scores(scores, labels):
    preds = torch.argmax(scores, dim=1)
    return (preds == labels).float().mean().item()


def train_standard_nn(
    data,
    width,
    epochs,
    lr,
    device,
):
    """
    Full-batch standard NN benchmark.

    This intentionally does NOT record detailed process timings.
    It only records accuracy/loss over epochs and epoch times.
    """

    model = NeuralNetwork(
        layer_sizes=[data["input_dim"], width, data["output_dim"]],
        activation=torch.relu,
    ).to(device)

    X_train = data["X_train"]
    y_train = data["y_train"].long()
    X_test = data["X_test"]
    y_test = data["y_test"].long()

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": [],
        "avg_epoch_time": [],
        "total_elapsed_time": [],
    }

    total_start = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        sync_if_cuda(device)
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(X_train)
        loss = loss_func(logits, y_train)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            test_logits = model(X_test)

            train_loss = loss_func(train_logits, y_train).item()
            test_loss = loss_func(test_logits, y_test).item()

            train_acc = classification_accuracy_from_scores(train_logits, y_train)
            test_acc = classification_accuracy_from_scores(test_logits, y_test)

        sync_if_cuda(device)
        epoch_time = time.time() - epoch_start
        total_elapsed_time = time.time() - total_start

        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)
        history["avg_epoch_time"].append(avg_epoch_time)
        history["total_elapsed_time"].append(total_elapsed_time)

        print(
            f"[standard_nn | width={width}] "
            f"Epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"epoch_time={epoch_time:.2f}s | "
            f"avg_epoch_time={avg_epoch_time:.2f}s"
        )

    total_training_time = time.time() - total_start

    summary = {
        "model": "standard_nn",
        "width": width,
        "epochs": epochs,
        "lr": lr,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "total_training_time": total_training_time,
    }

    return history, summary


def make_rvfl_model(
    data,
    width,
    lamb,
    seed,
    device,
    dtype,
    ortho_method,
):
    layers_info = [
        {
            "layer_dim": width,
            "weight_dist": "normal",
            "weight_var": 1.0,
            "gamma_k": None,
            "bias_switch": False,
            "bias_dist": "normal",
            "bias_var": 1.0,
            "unitary_init": "identity",
        }
    ]

    general_info = {
        "seed": seed,
        "device": device,
        "dtype": dtype,
    }

    model = RVFL(
        layersInfo=layers_info,
        generalInfo=general_info,
        orthoMatMethod=ortho_method,
        activation="relu",
        linkOption="direct",
        lamb=lamb,
        scalings=[0.5],
        input_dim=data["input_dim"],
        output_dim=data["output_dim"],
        task="classification",
    )

    return model


def train_rvfl_profiled(
    data,
    width,
    epochs,
    lr,
    lamb,
    seed,
    device,
    dtype,
    ortho_method,
):
    """
    RVFL benchmark.

    Timing columns:
        forward_time
        solve_beta_time
        pred_loss_time
        backward_time
        optimizer_time
        eval_time
        other_time

    Accuracy/loss are evaluated using the same beta solved before the optimizer step.
    This keeps exactly one beta solve per epoch.
    """

    total_start = time.time()

    model = make_rvfl_model(
        data=data,
        width=width,
        lamb=lamb,
        seed=seed,
        device=device,
        dtype=dtype,
        ortho_method=ortho_method,
    )

    X_train = data["X_train"]
    y_train_labels = data["y_train"].long()
    X_test = data["X_test"]
    y_test_labels = data["y_test"].long()

    X_train = model._as_feature_tensor(X_train)
    X_test = model._as_feature_tensor(X_test)

    Y_train, _ = model._prepare_targets(y_train_labels, fit_output_dim=True)
    Y_test, _ = model._prepare_targets(y_test_labels, fit_output_dim=False)

    sync_if_cuda(device)
    setup_start = time.time()

    model.create_hidden_layers(input_dim=data["input_dim"])

    if len(model.internal_layers) == 1:
        Z_train = model.precompute_one_layer_XW(X_train)
        Z_test = model.precompute_one_layer_XW(X_test)
    else:
        Z_train = None
        Z_test = None

    sync_if_cuda(device)
    setup_time = time.time() - setup_start

    optimizer = torch.optim.Adam(model.unitaryParams.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": [],
        "avg_epoch_time": [],
        "total_elapsed_time": [],
        "forward_time": [],
        "solve_beta_time": [],
        "pred_loss_time": [],
        "backward_time": [],
        "optimizer_time": [],
        "eval_time": [],
        "other_time": [],
    }

    epoch_times = []

    for epoch in range(1, epochs + 1):
        sync_if_cuda(device)
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Training forward.
        sync_if_cuda(device)
        t0 = time.time()
        Phi = model.forward(X_train, Z_train)
        sync_if_cuda(device)
        forward_time = time.time() - t0

        # Solve beta for the current Q.
        sync_if_cuda(device)
        t0 = time.time()
        with torch.no_grad():
            beta = model.solve_beta(Phi.detach(), Y_train)
        sync_if_cuda(device)
        solve_beta_time = time.time() - t0

        # Training loss for gradient step.
        sync_if_cuda(device)
        t0 = time.time()
        train_scores_for_grad = Phi @ beta
        loss = F.mse_loss(train_scores_for_grad, Y_train)
        sync_if_cuda(device)
        pred_loss_time = time.time() - t0

        # Evaluate using the SAME beta already solved this epoch.
        # No second beta solve.
        sync_if_cuda(device)
        t0 = time.time()
        with torch.no_grad():
            train_loss = loss.item()
            train_acc = classification_accuracy_from_scores(
                train_scores_for_grad,
                y_train_labels,
            )

            Phi_test = model.forward(X_test, Z_test)
            test_scores = Phi_test @ beta
            test_loss = F.mse_loss(test_scores, Y_test).item()
            test_acc = classification_accuracy_from_scores(
                test_scores,
                y_test_labels,
            )

            model.beta = beta

        sync_if_cuda(device)
        eval_time = time.time() - t0

        # Backprop.
        sync_if_cuda(device)
        t0 = time.time()
        loss.backward()
        sync_if_cuda(device)
        backward_time = time.time() - t0

        # Optimizer step and projection.
        sync_if_cuda(device)
        t0 = time.time()
        optimizer.step()

        for U in model.unitaryParams:
            if hasattr(U, "project"):
                U.project()

        sync_if_cuda(device)
        optimizer_time = time.time() - t0

        sync_if_cuda(device)
        epoch_time = time.time() - epoch_start
        total_elapsed_time = time.time() - total_start

        measured_time = (
            forward_time
            + solve_beta_time
            + pred_loss_time
            + backward_time
            + optimizer_time
            + eval_time
        )

        other_time = max(epoch_time - measured_time, 0.0)

        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)
        history["avg_epoch_time"].append(avg_epoch_time)
        history["total_elapsed_time"].append(total_elapsed_time)
        history["forward_time"].append(forward_time)
        history["solve_beta_time"].append(solve_beta_time)
        history["pred_loss_time"].append(pred_loss_time)
        history["backward_time"].append(backward_time)
        history["optimizer_time"].append(optimizer_time)
        history["eval_time"].append(eval_time)
        history["other_time"].append(other_time)

        print(
            f"[rvfl | width={width}] "
            f"Epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"epoch_time={epoch_time:.2f}s | "
            f"avg_epoch_time={avg_epoch_time:.2f}s"
        )

        print(
            f"Profile | "
            f"forward={forward_time:.3f}s | "
            f"solve_beta={solve_beta_time:.3f}s | "
            f"pred_loss={pred_loss_time:.3f}s | "
            f"backward={backward_time:.3f}s | "
            f"optimizer={optimizer_time:.3f}s | "
            f"eval={eval_time:.3f}s | "
            f"other={other_time:.3f}s"
        )

    total_training_time = time.time() - total_start

    summary = {
        "model": "rvfl",
        "width": width,
        "epochs": epochs,
        "lr": lr,
        "lambda": lamb,
        "ortho_method": ortho_method,
        "setup_time": setup_time,
        "final_eval_time": 0.0,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "total_training_time": total_training_time,
    }

    return history, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "fashionmnist", "cifar10"],
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["rvfl", "standard_nn"],
    )

    parser.add_argument("--width", required=True, type=int)

    parser.add_argument("--rvfl-epochs", type=int, default=20)
    parser.add_argument("--nn-epochs", type=int, default=200)

    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--lamb", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
    )

    parser.add_argument(
        "--ortho-method",
        default="qr",
        choices=["qr", "givens"],
    )

    parser.add_argument(
        "--result-root",
        default="benchmark_results",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You requested --device cuda, but CUDA is not available.")
        device = "cuda"
    else:
        device = "cpu"

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Width: {args.width}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"LR: {args.lr}")
    print(f"Seed: {args.seed}")
    if args.model == "rvfl":
        print(f"Lambda: {args.lamb}")
        print(f"Orthogonal method: {args.ortho_method}")
    print("=" * 80)

    data = load_dataset(
        dataset_name=args.dataset,
        device=device,
        dtype=dtype,
    )

    data = apply_debug_limits(
        data,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    if args.model == "standard_nn":
        epochs = args.nn_epochs

        history, summary = train_standard_nn(
            data=data,
            width=args.width,
            epochs=epochs,
            lr=args.lr,
            device=device,
        )

    elif args.model == "rvfl":
        epochs = args.rvfl_epochs

        history, summary = train_rvfl_profiled(
            data=data,
            width=args.width,
            epochs=epochs,
            lr=args.lr,
            lamb=args.lamb,
            seed=args.seed,
            device=device,
            dtype=dtype,
            ortho_method=args.ortho_method,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    out_dir = (
        Path(args.result_root)
        / args.dataset
        / args.model
        / f"width_{args.width}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"
    summary_path = out_dir / "summary.json"

    summary.update(
        {
            "dataset": args.dataset,
            "device": device,
            "dtype": str(dtype),
            "seed": args.seed,
            "input_dim": data["input_dim"],
            "output_dim": data["output_dim"],
            "history_file": str(history_path),
        }
    )

    save_history_csv(history, history_path)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Finished experiment.")
    print(f"History saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()