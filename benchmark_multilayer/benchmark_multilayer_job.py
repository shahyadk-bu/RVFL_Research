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

# Keep CUDA float32 behavior as close as practical to CPU float32.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

THIS_FILE = Path(__file__).resolve()
INNER_REPO = THIS_FILE.parents[1]
OUTER_REPO = INNER_REPO.parent
sys.path.insert(0, str(INNER_REPO))
sys.path.insert(0, str(OUTER_REPO))

from Regular_Neural_Network.model import NeuralNetwork
from RVFL_Research.Unitary_Model.Model.RVFL_Model import RVFL
from RVFL_Research.Unitary_Model.Model.utils import (
    load_cifar10_tensors,
    load_fashion_mnist_tensors,
    load_mnist_tensors,
)


def sync_if_cuda(device):
    is_cuda = device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(text):
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list such as 200,200,100.")
    try:
        parsed = [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("All widths must be integers.") from exc
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("All widths must be positive.")
    return parsed


def parse_float_list(text):
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a float or comma-separated float list.")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("All scaling values must be numeric.") from exc


def expand_scalings(raw_scalings, depth):
    if len(raw_scalings) == 1:
        return raw_scalings * depth
    if len(raw_scalings) != depth:
        raise ValueError(
            f"Received {len(raw_scalings)} scaling values for depth {depth}. "
            "Pass one value to repeat, or one value per layer."
        )
    return raw_scalings


def float_tag(value):
    return f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")


def architecture_tag(widths):
    return "x".join(str(width) for width in widths)


def scaling_tag(scalings):
    return "-".join(float_tag(value) for value in scalings)


def save_history_csv(history, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    n_rows = len(history[keys[0]])
    with filepath.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for index in range(n_rows):
            writer.writerow([history[key][index] for key in keys])


def load_dataset(dataset_name, device, dtype):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        X_train, _, X_test, _, y_train, y_test = load_mnist_tensors(
            device=device,
            dtype=dtype,
        )
        return {
            "X_train": X_train,
            "y_train": y_train.long(),
            "X_test": X_test,
            "y_test": y_test.long(),
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
    if train_limit is not None:
        data["X_train"] = data["X_train"][:train_limit]
        data["y_train"] = data["y_train"][:train_limit]
    if test_limit is not None:
        data["X_test"] = data["X_test"][:test_limit]
        data["y_test"] = data["y_test"][:test_limit]
    return data


@torch.no_grad()
def classification_accuracy_from_scores(scores, labels):
    return (torch.argmax(scores, dim=1) == labels).float().mean().item()


def architecture_metadata(data, widths, link_option, dtype):
    depth = len(widths)
    input_dim = int(data["input_dim"])
    n_train = int(data["X_train"].shape[0])

    random_weight_count = input_dim * widths[0]
    random_weight_count += sum(widths[i - 1] * widths[i] for i in range(1, depth))
    unitary_parameter_count = sum(width * width for width in widths)

    if link_option == "multi":
        design_dim = input_dim + sum(widths)
    elif link_option == "direct":
        design_dim = input_dim + widths[-1]
    elif link_option == "none":
        design_dim = widths[-1]
    else:
        raise ValueError(f"Invalid link option: {link_option}")

    bytes_per_value = torch.tensor([], dtype=dtype).element_size()
    ridge_system_dim = design_dim if design_dim <= n_train else n_train

    return {
        "architecture": architecture_tag(widths),
        "widths": widths,
        "depth": depth,
        "constant_width": len(set(widths)) == 1,
        "total_hidden_units": sum(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "random_weight_count": random_weight_count,
        "unitary_parameter_count": unitary_parameter_count,
        "design_dim": design_dim,
        "ridge_system_dim": ridge_system_dim,
        "estimated_design_matrix_gb": n_train * design_dim * bytes_per_value / 1e9,
        "estimated_ridge_matrix_gb": ridge_system_dim * ridge_system_dim * bytes_per_value / 1e9,
        "qr_cost_proxy": sum(width ** 3 for width in widths),
    }


def train_standard_nn(data, widths, epochs, lr, device):
    model = NeuralNetwork(
        layer_sizes=[data["input_dim"], *widths, data["output_dim"]],
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
    arch = architecture_tag(widths)

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
        total_elapsed = time.time() - total_start
        epoch_times.append(epoch_time)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)
        history["avg_epoch_time"].append(sum(epoch_times) / len(epoch_times))
        history["total_elapsed_time"].append(total_elapsed)

        print(
            f"[standard_nn | arch={arch}] Epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | "
            f"epoch_time={epoch_time:.3f}s"
        )

    return history, {
        "model": "standard_nn",
        "epochs": epochs,
        "lr": lr,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "total_training_time": time.time() - total_start,
    }


def make_rvfl_model(data, widths, scalings, lamb, seed, device, dtype, ortho_method, link_option):
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
        for width in widths
    ]

    return RVFL(
        layersInfo=layers_info,
        generalInfo={"seed": seed, "device": device, "dtype": dtype},
        orthoMatMethod=ortho_method,
        activation="relu",
        linkOption=link_option,
        lamb=lamb,
        scalings=scalings,
        input_dim=data["input_dim"],
        output_dim=data["output_dim"],
        task="classification",
    )


def train_rvfl_profiled(
    data,
    widths,
    scalings,
    epochs,
    lr,
    lamb,
    seed,
    device,
    dtype,
    ortho_method,
    link_option,
):
    total_start = time.time()
    model = make_rvfl_model(
        data=data,
        widths=widths,
        scalings=scalings,
        lamb=lamb,
        seed=seed,
        device=device,
        dtype=dtype,
        ortho_method=ortho_method,
        link_option=link_option,
    )

    X_train = model._as_feature_tensor(data["X_train"])
    X_test = model._as_feature_tensor(data["X_test"])
    y_train_labels = data["y_train"].long()
    y_test_labels = data["y_test"].long()
    Y_train, _ = model._prepare_targets(y_train_labels, fit_output_dim=True)
    Y_test, _ = model._prepare_targets(y_test_labels, fit_output_dim=False)

    sync_if_cuda(device)
    setup_start = time.time()
    model.create_hidden_layers(input_dim=data["input_dim"])

    # The exact XW shortcut only applies to one hidden layer.
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
        "train_step_time": [],
        "avg_train_step_time": [],
        "eval_total_time": [],
        "avg_eval_total_time": [],
        "total_elapsed_time": [],
        "train_forward_time": [],
        "train_solve_beta_time": [],
        "train_pred_loss_time": [],
        "backward_time": [],
        "adam_step_time": [],
        "projection_time": [],
        "optimizer_time": [],
        "train_other_time": [],
        "eval_forward_time": [],
        "eval_solve_beta_time": [],
        "eval_pred_time": [],
        # Legacy aliases retained for compatibility with prior analysis scripts.
        "forward_time": [],
        "solve_beta_time": [],
        "pred_loss_time": [],
        "eval_time": [],
        "other_time": [],
    }

    epoch_times = []
    train_step_times = []
    eval_times = []
    arch = architecture_tag(widths)

    for epoch in range(1, epochs + 1):
        sync_if_cuda(device)
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sync_if_cuda(device)
        t0 = time.time()
        Phi_train = model.forward(X_train, Z_train)
        sync_if_cuda(device)
        train_forward_time = time.time() - t0

        sync_if_cuda(device)
        t0 = time.time()
        with torch.no_grad():
            beta_train = model.solve_beta(Phi_train.detach(), Y_train)
        sync_if_cuda(device)
        train_solve_beta_time = time.time() - t0

        sync_if_cuda(device)
        t0 = time.time()
        train_scores_for_grad = Phi_train @ beta_train
        loss = F.mse_loss(train_scores_for_grad, Y_train)
        sync_if_cuda(device)
        train_pred_loss_time = time.time() - t0

        sync_if_cuda(device)
        t0 = time.time()
        loss.backward()
        sync_if_cuda(device)
        backward_time = time.time() - t0

        sync_if_cuda(device)
        t0 = time.time()
        optimizer.step()
        sync_if_cuda(device)
        adam_step_time = time.time() - t0

        sync_if_cuda(device)
        t0 = time.time()
        for unitary in model.unitaryParams:
            if hasattr(unitary, "project"):
                unitary.project()
        sync_if_cuda(device)
        projection_time = time.time() - t0
        optimizer_time = adam_step_time + projection_time

        model.eval()
        sync_if_cuda(device)
        eval_start = time.time()
        with torch.no_grad():
            sync_if_cuda(device)
            t0 = time.time()
            Phi_train_eval = model.forward(X_train, Z_train)
            Phi_test_eval = model.forward(X_test, Z_test)
            sync_if_cuda(device)
            eval_forward_time = time.time() - t0

            sync_if_cuda(device)
            t0 = time.time()
            beta_eval = model.solve_beta(Phi_train_eval, Y_train)
            sync_if_cuda(device)
            eval_solve_beta_time = time.time() - t0

            sync_if_cuda(device)
            t0 = time.time()
            train_scores = Phi_train_eval @ beta_eval
            test_scores = Phi_test_eval @ beta_eval
            train_loss = F.mse_loss(train_scores, Y_train).item()
            test_loss = F.mse_loss(test_scores, Y_test).item()
            train_acc = classification_accuracy_from_scores(train_scores, y_train_labels)
            test_acc = classification_accuracy_from_scores(test_scores, y_test_labels)
            model.beta = beta_eval
            sync_if_cuda(device)
            eval_pred_time = time.time() - t0

        sync_if_cuda(device)
        eval_total_time = time.time() - eval_start
        epoch_time = time.time() - epoch_start
        total_elapsed_time = time.time() - total_start

        train_components = (
            train_forward_time
            + train_solve_beta_time
            + train_pred_loss_time
            + backward_time
            + adam_step_time
            + projection_time
        )
        measured = train_components + eval_forward_time + eval_solve_beta_time + eval_pred_time
        other_time = max(epoch_time - measured, 0.0)
        train_step_time = max(epoch_time - eval_total_time, 0.0)
        train_other_time = max(train_step_time - train_components, 0.0)

        epoch_times.append(epoch_time)
        train_step_times.append(train_step_time)
        eval_times.append(eval_total_time)

        values = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time,
            "avg_epoch_time": sum(epoch_times) / len(epoch_times),
            "train_step_time": train_step_time,
            "avg_train_step_time": sum(train_step_times) / len(train_step_times),
            "eval_total_time": eval_total_time,
            "avg_eval_total_time": sum(eval_times) / len(eval_times),
            "total_elapsed_time": total_elapsed_time,
            "train_forward_time": train_forward_time,
            "train_solve_beta_time": train_solve_beta_time,
            "train_pred_loss_time": train_pred_loss_time,
            "backward_time": backward_time,
            "adam_step_time": adam_step_time,
            "projection_time": projection_time,
            "optimizer_time": optimizer_time,
            "train_other_time": train_other_time,
            "eval_forward_time": eval_forward_time,
            "eval_solve_beta_time": eval_solve_beta_time,
            "eval_pred_time": eval_pred_time,
            "forward_time": train_forward_time,
            "solve_beta_time": train_solve_beta_time,
            "pred_loss_time": train_pred_loss_time,
            "eval_time": eval_total_time,
            "other_time": other_time,
        }
        for key, value in values.items():
            history[key].append(value)

        print(
            f"[rvfl | arch={arch} | link={link_option}] Epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | "
            f"wall={epoch_time:.3f}s | train={train_step_time:.3f}s | eval={eval_total_time:.3f}s"
        )
        print(
            "Train profile | "
            f"forward={train_forward_time:.3f}s | solve_beta={train_solve_beta_time:.3f}s | "
            f"pred_loss={train_pred_loss_time:.3f}s | backward={backward_time:.3f}s | "
            f"adam={adam_step_time:.3f}s | projection={projection_time:.3f}s | "
            f"other={train_other_time:.3f}s"
        )

    summary = {
        "model": "rvfl",
        "epochs": epochs,
        "lr": lr,
        "lambda": lamb,
        "ortho_method": ortho_method,
        "link_option": link_option,
        "scalings": scalings,
        "setup_time": setup_time,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "total_training_time": time.time() - total_start,
        "avg_train_step_time": sum(history["train_step_time"]) / len(history["train_step_time"]),
        "total_train_step_time": sum(history["train_step_time"]),
        "avg_eval_total_time": sum(history["eval_total_time"]) / len(history["eval_total_time"]),
        "total_eval_time": sum(history["eval_total_time"]),
        "avg_eval_solve_beta_time": sum(history["eval_solve_beta_time"]) / len(history["eval_solve_beta_time"]),
        "avg_train_forward_time": sum(history["train_forward_time"]) / len(history["train_forward_time"]),
        "avg_train_solve_beta_time": sum(history["train_solve_beta_time"]) / len(history["train_solve_beta_time"]),
        "avg_train_pred_loss_time": sum(history["train_pred_loss_time"]) / len(history["train_pred_loss_time"]),
        "avg_backward_time": sum(history["backward_time"]) / len(history["backward_time"]),
        "avg_adam_step_time": sum(history["adam_step_time"]) / len(history["adam_step_time"]),
        "avg_projection_time": sum(history["projection_time"]) / len(history["projection_time"]),
        "avg_optimizer_time": sum(history["optimizer_time"]) / len(history["optimizer_time"]),
    }
    return history, summary


def build_output_dir(result_root, args, widths, scalings):
    base = Path(result_root) / args.dataset / args.model
    arch = f"arch_{architecture_tag(widths)}"
    seed = f"seed_{args.seed}"
    lr = f"lr_{float_tag(args.lr)}"

    if args.model == "standard_nn":
        return base / arch / lr / seed

    return (
        base
        / f"link_{args.link_option}"
        / arch
        / lr
        / f"lam_{float_tag(args.lamb)}"
        / f"scale_{scaling_tag(scalings)}"
        / f"ortho_{args.ortho_method}"
        / seed
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashionmnist", "cifar10"])
    parser.add_argument("--model", required=True, choices=["rvfl", "standard_nn"])
    parser.add_argument("--widths", required=True, type=parse_int_list)
    parser.add_argument("--scalings", type=parse_float_list, default=[0.0])
    parser.add_argument("--link-option", default="multi", choices=["none", "direct", "multi"])
    parser.add_argument("--architecture-family", default="unspecified")
    parser.add_argument("--rvfl-epochs", type=int, default=20)
    parser.add_argument("--nn-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lamb", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--ortho-method", default="qr", choices=["qr", "givens"])
    parser.add_argument("--result-root", default="benchmark_multilayer/results")
    args = parser.parse_args()

    widths = args.widths
    scalings = expand_scalings(args.scalings, len(widths))
    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = "cuda"
    else:
        device = "cpu"

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print("=" * 90)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Architecture: {architecture_tag(widths)}")
    print(f"Depth: {len(widths)}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"LR: {args.lr}")
    print(f"Seed: {args.seed}")
    if args.model == "rvfl":
        print(f"Lambda: {args.lamb}")
        print(f"Link option: {args.link_option}")
        print(f"Scalings: {scalings}")
        print(f"Orthogonal method: {args.ortho_method}")
    print("=" * 90)

    data = load_dataset(args.dataset, device, dtype)
    data = apply_debug_limits(data, args.train_limit, args.test_limit)

    metadata_link = args.link_option if args.model == "rvfl" else "none"
    metadata = architecture_metadata(data, widths, metadata_link, dtype)
    print(
        f"Design dimension estimate: {metadata['design_dim']} | "
        f"ridge matrix estimate: {metadata['estimated_ridge_matrix_gb']:.3f} GB | "
        f"unitary entries: {metadata['unitary_parameter_count']:,}"
    )

    if args.model == "standard_nn":
        history, summary = train_standard_nn(data, widths, args.nn_epochs, args.lr, device)
    else:
        history, summary = train_rvfl_profiled(
            data=data,
            widths=widths,
            scalings=scalings,
            epochs=args.rvfl_epochs,
            lr=args.lr,
            lamb=args.lamb,
            seed=args.seed,
            device=device,
            dtype=dtype,
            ortho_method=args.ortho_method,
            link_option=args.link_option,
        )

    out_dir = build_output_dir(args.result_root, args, widths, scalings)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"
    summary_path = out_dir / "summary.json"

    summary.update(metadata)
    summary.update(
        {
            "dataset": args.dataset,
            "device": device,
            "dtype": str(dtype),
            "seed": args.seed,
            "input_dim": data["input_dim"],
            "output_dim": data["output_dim"],
            "n_train": int(data["X_train"].shape[0]),
            "n_test": int(data["X_test"].shape[0]),
            "architecture_family": args.architecture_family,
            "history_file": str(history_path),
        }
    )

    save_history_csv(history, history_path)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 90)
    print("Finished experiment.")
    print(f"History saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
