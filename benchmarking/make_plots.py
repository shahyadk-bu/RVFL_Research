import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import argparse


def load_all_results(result_root):
    result_root = Path(result_root)

    summaries = []
    histories = []

    for summary_path in result_root.rglob("summary.json"):
        if "plots" in summary_path.parts:
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        history_path = Path(summary["history_file"])

        if not history_path.exists():
            history_path = summary_path.parent / "history.csv"

        if not history_path.exists():
            print(f"Missing history file for: {summary_path}")
            continue

        history = pd.read_csv(history_path)

        history["dataset"] = summary.get("dataset")
        history["model"] = summary.get("model")
        history["width"] = summary.get("width")
        history["lr"] = summary.get("lr")
        history["seed"] = summary.get("seed")

        summaries.append(summary)
        histories.append(history)

    summary_df = pd.DataFrame(summaries)

    if histories:
        history_df = pd.concat(histories, ignore_index=True)
    else:
        history_df = pd.DataFrame()

    return summary_df, history_df


def lr_to_tag(lr):
    return str(lr).replace(".", "p").replace("-", "m")


def plot_individual_accuracy_curves(history_df, plot_root):
    out_dir = Path(plot_root) / "accuracy_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        print("No history data found. Skipping accuracy plots.")
        return

    groups = history_df.groupby(["dataset", "model", "width", "lr"])

    for (dataset, model, width, lr), df in groups:
        df = df.sort_values("epoch")

        plt.figure(figsize=(8, 5))

        plt.plot(
            df["epoch"],
            df["train_acc"],
            marker="o",
            label="train accuracy",
        )

        plt.plot(
            df["epoch"],
            df["test_acc"],
            marker="o",
            label="test accuracy",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model} on {dataset}, width={width}, lr={lr:g}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = (
            f"accuracy_{dataset}_{model}_width{width}_lr{lr_to_tag(lr)}.png"
        )
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def plot_test_accuracy_by_width(history_df, plot_root):
    """
    For each dataset/model/lr triple, plot test accuracy vs epoch for every width.
    """

    out_dir = Path(plot_root) / "test_accuracy_by_width"
    out_dir.mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        return

    groups = history_df.groupby(["dataset", "model", "lr"])

    for (dataset, model, lr), df_group in groups:
        plt.figure(figsize=(9, 6))

        for width, df in df_group.groupby("width"):
            df = df.sort_values("epoch")
            plt.plot(
                df["epoch"],
                df["test_acc"],
                marker="o",
                label=f"width={width}",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Test accuracy")
        plt.title(f"Test accuracy by width: {model} on {dataset}, lr={lr:g}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = (
            f"test_accuracy_by_width_{dataset}_{model}_lr{lr_to_tag(lr)}.png"
        )
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def plot_rvfl_time_breakdowns(history_df, plot_root):
    """
    RVFL-only timing plots.

    Produces:
        1. Training-step breakdown, excluding evaluation.
        2. Evaluation-only breakdown.
        3. Wall time vs train-step/eval time.
    """

    out_dir = Path(plot_root) / "rvfl_time_breakdowns"
    out_dir.mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        print("No history data found. Skipping RVFL timing plots.")
        return

    rvfl_df = history_df[history_df["model"] == "rvfl"].copy()

    if rvfl_df.empty:
        print("No RVFL data found. Skipping RVFL timing plots.")
        return

    # ------------------------------------------------------------
    # 1. Training-step breakdown.
    # ------------------------------------------------------------
    train_timing_cols = [
        "train_forward_time",
        "train_solve_beta_time",
        "train_pred_loss_time",
        "backward_time",
        "optimizer_time",
        "train_other_time",
    ]

    legacy_train_timing_cols = [
        "forward_time",
        "solve_beta_time",
        "pred_loss_time",
        "backward_time",
        "optimizer_time",
        "other_time",
    ]

    if all(c in rvfl_df.columns for c in train_timing_cols):
        train_cols = train_timing_cols
    else:
        train_cols = [c for c in legacy_train_timing_cols if c in rvfl_df.columns]

    if train_cols:
        for (dataset, lr), df_dataset in rvfl_df.groupby(["dataset", "lr"]):
            avg_times = (
                df_dataset.groupby("width")[train_cols]
                .mean()
                .sort_index()
            )

            ax = avg_times.plot(
                kind="bar",
                stacked=True,
                figsize=(10, 6),
            )

            ax.set_xlabel("Width")
            ax.set_ylabel("Average train-step time per epoch, seconds")
            ax.set_title(f"RVFL train-step time breakdown on {dataset}, lr={lr}")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()

            lr_tag = str(lr).replace(".", "p").replace("-", "m")
            filename = f"rvfl_train_step_breakdown_{dataset}_lr{lr_tag}.png"
            plt.savefig(out_dir / filename, dpi=200)
            plt.close()

    # ------------------------------------------------------------
    # 2. Evaluation-only breakdown.
    # ------------------------------------------------------------
    eval_timing_cols = [
        "eval_forward_time",
        "eval_solve_beta_time",
        "eval_pred_time",
    ]

    if all(c in rvfl_df.columns for c in eval_timing_cols):
        for (dataset, lr), df_dataset in rvfl_df.groupby(["dataset", "lr"]):
            avg_times = (
                df_dataset.groupby("width")[eval_timing_cols]
                .mean()
                .sort_index()
            )

            ax = avg_times.plot(
                kind="bar",
                stacked=True,
                figsize=(10, 6),
            )

            ax.set_xlabel("Width")
            ax.set_ylabel("Average evaluation-only time per epoch, seconds")
            ax.set_title(f"RVFL evaluation-only breakdown on {dataset}, lr={lr}")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()

            lr_tag = str(lr).replace(".", "p").replace("-", "m")
            filename = f"rvfl_eval_breakdown_{dataset}_lr{lr_tag}.png"
            plt.savefig(out_dir / filename, dpi=200)
            plt.close()

    # ------------------------------------------------------------
    # 3. Full wall time vs training-only/eval time.
    # ------------------------------------------------------------
    wall_compare_cols = [
        "epoch_time",
        "train_step_time",
        "eval_total_time",
    ]

    if all(c in rvfl_df.columns for c in wall_compare_cols):
        for (dataset, lr), df_dataset in rvfl_df.groupby(["dataset", "lr"]):
            avg_times = (
                df_dataset.groupby("width")[wall_compare_cols]
                .mean()
                .sort_index()
            )

            ax = avg_times.plot(
                kind="bar",
                figsize=(10, 6),
            )

            ax.set_xlabel("Width")
            ax.set_ylabel("Average time per epoch, seconds")
            ax.set_title(f"RVFL wall vs train/eval time on {dataset}, lr={lr}")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()

            lr_tag = str(lr).replace(".", "p").replace("-", "m")
            filename = f"rvfl_wall_vs_train_eval_{dataset}_lr{lr_tag}.png"
            plt.savefig(out_dir / filename, dpi=200)
            plt.close()


def save_tables(summary_df, history_df, result_root):
    result_root = Path(result_root)

    summary_all_path = result_root / "summary_all.csv"
    history_all_path = result_root / "history_all.csv"
    time_table_path = result_root / "time_table.csv"
    epoch_time_table_path = result_root / "epoch_time_table.csv"

    summary_df.to_csv(summary_all_path, index=False)
    history_df.to_csv(history_all_path, index=False)

    time_table_cols = [
        "dataset",
        "model",
        "width",
        "epochs",
        "lr",
        "lambda",
        "ortho_method",
        "input_dim",
        "output_dim",

        "setup_time",

        # Full benchmark wall-clock timing.
        "avg_epoch_time",
        "total_training_time",

        # Training-only timing.
        "avg_train_step_time",
        "total_train_step_time",

        # Evaluation-only timing.
        "avg_eval_total_time",
        "total_eval_time",
        "avg_eval_solve_beta_time",

        # RVFL component averages.
        "avg_train_forward_time",
        "avg_train_solve_beta_time",
        "avg_train_pred_loss_time",
        "avg_backward_time",
        "avg_optimizer_time",

        # Final performance.
        "final_train_acc",
        "final_test_acc",
        "final_train_loss",
        "final_test_loss",

        "seed",
        "device",
        "dtype",
        "history_file",
    ]

    existing_time_cols = [c for c in time_table_cols if c in summary_df.columns]

    if existing_time_cols:
        time_table = summary_df[existing_time_cols].sort_values(
            ["dataset", "model", "width"]
        )
        time_table.to_csv(time_table_path, index=False)

    epoch_time_cols = [
        "dataset",
        "model",
        "width",
        "epoch",
        "lr",
        "seed",

        "train_acc",
        "test_acc",
        "train_loss",
        "test_loss",

        # General timing.
        "epoch_time",
        "avg_epoch_time",
        "train_step_time",
        "avg_train_step_time",
        "eval_total_time",
        "avg_eval_total_time",
        "total_elapsed_time",

        # RVFL train-step timing.
        "train_forward_time",
        "train_solve_beta_time",
        "train_pred_loss_time",
        "backward_time",
        "optimizer_time",
        "train_other_time",

        # RVFL evaluation-only timing.
        "eval_forward_time",
        "eval_solve_beta_time",
        "eval_pred_time",

        # Legacy aliases.
        "forward_time",
        "solve_beta_time",
        "pred_loss_time",
        "eval_time",
        "other_time",
    ]

    existing_epoch_cols = [c for c in epoch_time_cols if c in history_df.columns]

    if existing_epoch_cols:
        epoch_time_table = history_df[existing_epoch_cols].sort_values(
            ["dataset", "model", "width", "epoch"]
        )
        epoch_time_table.to_csv(epoch_time_table_path, index=False)

    print(f"Saved combined summary to: {summary_all_path}")
    print(f"Saved combined history to: {history_all_path}")
    print(f"Saved time table to: {time_table_path}")
    print(f"Saved epoch time table to: {epoch_time_table_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="benchmark_results")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    plot_root = result_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    summary_df, history_df = load_all_results(result_root)

    if summary_df.empty:
        print("No summary files found. Did the benchmark jobs finish?")
        return

    save_tables(summary_df, history_df, result_root)

    plot_individual_accuracy_curves(history_df, plot_root)
    plot_test_accuracy_by_width(history_df, plot_root)
    plot_rvfl_time_breakdowns(history_df, plot_root)

    print(f"Saved plots to: {plot_root}")


if __name__ == "__main__":
    main()