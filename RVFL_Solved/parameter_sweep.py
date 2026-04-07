import os
import csv
import itertools
import torch

from RVFL_Model import RVFL
from utils import load_mnist_tensors, accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    seed = 0

    generalInfo = {
        "seed": seed,
        "device": device,
        "dtype": dtype
    }

    X_train, y_train, X_test, y_test, y_train_labels, y_test_labels = load_mnist_tensors(
        device=device,
        dtype=dtype
    )

    link_options = ["none", "direct"]
    activations = ["relu", "sigmoid", "tanh"]
    layer_sizes = [50, 100, 250, 500, 1000]
    weight_vars = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0]
    scalings = [0.0, 0.25, 0.5, 0.75, 1.0]
    lambdas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]

    weight_dist = "normal"
    bias_switch = False
    bias_dist = "normal"
    bias_var = 1.0
    gamma_k = None

    # Save setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "sweep_results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "parameter_sweep_results.csv")

    fieldnames = [
        "status",
        "seed",
        "activation",
        "linkOption",
        "layer_dim",
        "weight_dist",
        "weight_var",
        "gamma_k",
        "bias_switch",
        "bias_dist",
        "bias_var",
        "scaling",
        "lambda",
        "train_acc",
        "test_acc",
        "model_filename",
        "error_message",
    ]

    total_runs = (
        len(link_options)
        * len(activations)
        * len(layer_sizes)
        * len(weight_vars)
        * len(scalings)
        * len(lambdas)
    )

    run_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for linkOption, activation, layer_dim, weight_var, scaling, lamb in itertools.product(
            link_options,
            activations,
            layer_sizes,
            weight_vars,
            scalings,
            lambdas
        ):
            run_count += 1
            print(f"Run {run_count}/{total_runs}")

            layersInfo = [
                {
                    "layer_dim": layer_dim,
                    "weight_dist": weight_dist,
                    "weight_var": weight_var,
                    "gamma_k": gamma_k,
                    "bias_switch": bias_switch,
                    "bias_dist": bias_dist,
                    "bias_var": bias_var,
                }
            ]

            try:
                model = RVFL(
                    layersInfo=layersInfo,
                    generalInfo=generalInfo,
                    activation=activation,
                    linkOption=linkOption,
                    lamb=lamb,
                    scalings=[scaling]
                )

                model.fit(X_train, y_train)

                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)

                train_acc = accuracy(train_preds, y_train_labels)
                test_acc = accuracy(test_preds, y_test_labels)

                model.saveModel()

                model_filename = (
                    f"RVFL"
                    f"_seed-{seed}"
                    f"_act-{activation}"
                    f"_link-{linkOption}"
                    f"_lam-{lamb:.0e}"
                    f"_dims-{layer_dim}"
                    f"_wdist-{weight_dist}"
                    f"_wvar-{weight_var}"
                    f"_bias-{bias_switch}"
                    f"_bdist-{bias_dist}"
                    f"_bvar-{bias_var}"
                    f"_scale-{scaling}.pt"
                )

                row = {
                    "status": "success",
                    "seed": seed,
                    "activation": activation,
                    "linkOption": linkOption,
                    "layer_dim": layer_dim,
                    "weight_dist": weight_dist,
                    "weight_var": weight_var,
                    "gamma_k": gamma_k,
                    "bias_switch": bias_switch,
                    "bias_dist": bias_dist,
                    "bias_var": bias_var,
                    "scaling": scaling,
                    "lambda": lamb,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "model_filename": model_filename,
                    "error_message": "",
                }

                print(
                    f"  done | act={activation}, link={linkOption}, dim={layer_dim}, "
                    f"var={weight_var}, scale={scaling}, lambda={lamb} | "
                    f"train={train_acc:.4f}, test={test_acc:.4f}"
                )

            except Exception as e:
                row = {
                    "status": "failed",
                    "seed": seed,
                    "activation": activation,
                    "linkOption": linkOption,
                    "layer_dim": layer_dim,
                    "weight_dist": weight_dist,
                    "weight_var": weight_var,
                    "gamma_k": gamma_k,
                    "bias_switch": bias_switch,
                    "bias_dist": bias_dist,
                    "bias_var": bias_var,
                    "scaling": scaling,
                    "lambda": lamb,
                    "train_acc": "",
                    "test_acc": "",
                    "model_filename": "",
                    "error_message": str(e),
                }

                print(
                    f"  failed | act={activation}, link={linkOption}, dim={layer_dim}, "
                    f"var={weight_var}, scale={scaling}, lambda={lamb}"
                )
                print(f"  error: {e}")

            writer.writerow(row)
            f.flush()

    print("\nSweep complete.")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()