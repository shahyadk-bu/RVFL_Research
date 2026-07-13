import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_tag(value):
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "_")


def load_all_results(result_root):
    result_root = Path(result_root)
    summaries = []
    histories = []

    for summary_path in result_root.rglob("summary.json"):
        if "plots" in summary_path.parts:
            continue

        with summary_path.open() as handle:
            summary = json.load(handle)

        history_path = Path(summary.get("history_file", ""))
        if not history_path.exists():
            history_path = summary_path.parent / "history.csv"
        if not history_path.exists():
            print(f"Missing history file for {summary_path}")
            continue

        history = pd.read_csv(history_path)
        for key in [
            "dataset",
            "model",
            "architecture",
            "architecture_family",
            "depth",
            "constant_width",
            "min_width",
            "max_width",
            "total_hidden_units",
            "design_dim",
            "unitary_parameter_count",
            "qr_cost_proxy",
            "lr",
            "lambda",
            "link_option",
            "seed",
        ]:
            history[key] = summary.get(key)

        summaries.append(summary)
        histories.append(history)

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return summary_df, history_df


def save_tables(summary_df, history_df, result_root):
    result_root = Path(result_root)
    summary_df.to_csv(result_root / "summary_all.csv", index=False)
    history_df.to_csv(result_root / "history_all.csv", index=False)

    ranking_cols = [
        "dataset",
        "model",
        "link_option",
        "architecture_family",
        "architecture",
        "depth",
        "seed",
        "lr",
        "lambda",
        "scalings",
        "total_hidden_units",
        "design_dim",
        "unitary_parameter_count",
        "qr_cost_proxy",
        "final_train_acc",
        "final_test_acc",
        "avg_epoch_time",
        "avg_train_step_time",
        "avg_projection_time",
        "total_training_time",
    ]
    existing = [column for column in ranking_cols if column in summary_df.columns]
    rankings = summary_df[existing].sort_values(
        [column for column in ["dataset", "model", "final_test_acc"] if column in existing],
        ascending=[True, True, False][: len([column for column in ["dataset", "model", "final_test_acc"] if column in existing])],
    )
    rankings.to_csv(result_root / "architecture_rankings.csv", index=False)

    aggregate_keys = [
        column
        for column in [
            "dataset",
            "model",
            "link_option",
            "architecture_family",
            "architecture",
            "depth",
            "lr",
            "lambda",
            "total_hidden_units",
            "design_dim",
            "unitary_parameter_count",
            "qr_cost_proxy",
        ]
        if column in summary_df.columns
    ]
    metric_cols = [
        column
        for column in [
            "final_train_acc",
            "final_test_acc",
            "avg_epoch_time",
            "avg_train_step_time",
            "avg_projection_time",
            "total_training_time",
        ]
        if column in summary_df.columns
    ]
    if aggregate_keys and metric_cols:
        aggregate = summary_df.groupby(aggregate_keys, dropna=False)[metric_cols].agg(["mean", "std", "count"])
        aggregate.columns = ["_".join(parts) for parts in aggregate.columns]
        aggregate.reset_index().to_csv(result_root / "architecture_aggregate.csv", index=False)


def plot_individual_accuracy_curves(history_df, plot_root):
    out_dir = Path(plot_root) / "accuracy_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        return

    group_cols = ["dataset", "model", "architecture", "lr", "seed", "link_option"]
    for keys, frame in history_df.groupby(group_cols, dropna=False):
        dataset, model, architecture, lr, seed, link_option = keys
        frame = frame.sort_values("epoch")
        plt.figure(figsize=(8, 5))
        plt.plot(frame["epoch"], frame["train_acc"], marker="o", label="train")
        plt.plot(frame["epoch"], frame["test_acc"], marker="o", label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{model} on {dataset}: {architecture}, lr={lr:g}, "
            f"seed={seed}, link={link_option}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = (
            f"accuracy_{dataset}_{model}_{architecture}_lr{safe_tag(lr)}_"
            f"seed{seed}_link{safe_tag(link_option)}.png"
        )
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def constant_width_subset(summary_df):
    if summary_df.empty or "constant_width" not in summary_df.columns:
        return pd.DataFrame()
    mask = summary_df["constant_width"].astype(str).str.lower().isin(["true", "1"])
    return summary_df[mask].copy()


def plot_accuracy_vs_depth(summary_df, plot_root):
    out_dir = Path(plot_root) / "accuracy_vs_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = constant_width_subset(summary_df)
    if frame.empty:
        return

    group_cols = ["dataset", "model", "lr", "link_option"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        dataset, model, lr, link_option = keys
        plt.figure(figsize=(9, 6))
        for width, width_group in group.groupby("max_width"):
            stats = width_group.groupby("depth")["final_test_acc"].agg(["mean", "std"]).sort_index()
            plt.plot(stats.index, stats["mean"], marker="o", label=f"width={int(width)}")
            if stats["std"].notna().any():
                lower = stats["mean"] - stats["std"].fillna(0)
                upper = stats["mean"] + stats["std"].fillna(0)
                plt.fill_between(stats.index, lower, upper, alpha=0.15)
        plt.xlabel("Number of hidden layers")
        plt.ylabel("Final test accuracy")
        plt.title(f"Accuracy vs depth: {model} on {dataset}, lr={lr:g}, link={link_option}")
        plt.xticks(sorted(group["depth"].unique()))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"accuracy_vs_depth_{dataset}_{model}_lr{safe_tag(lr)}_link{safe_tag(link_option)}.png",
            dpi=200,
        )
        plt.close()


def plot_time_vs_depth(summary_df, plot_root):
    out_dir = Path(plot_root) / "time_vs_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = constant_width_subset(summary_df)
    if frame.empty:
        return

    time_column = "avg_train_step_time" if "avg_train_step_time" in frame.columns else "avg_epoch_time"
    group_cols = ["dataset", "model", "lr", "link_option"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        dataset, model, lr, link_option = keys
        plt.figure(figsize=(9, 6))
        for width, width_group in group.groupby("max_width"):
            stats = width_group.groupby("depth")[time_column].agg(["mean", "std"]).sort_index()
            plt.plot(stats.index, stats["mean"], marker="o", label=f"width={int(width)}")
            if stats["std"].notna().any():
                lower = stats["mean"] - stats["std"].fillna(0)
                upper = stats["mean"] + stats["std"].fillna(0)
                plt.fill_between(stats.index, lower, upper, alpha=0.15)
        plt.xlabel("Number of hidden layers")
        plt.ylabel(f"{time_column.replace('_', ' ')} (seconds)")
        plt.title(f"Time vs depth: {model} on {dataset}, lr={lr:g}, link={link_option}")
        plt.xticks(sorted(group["depth"].unique()))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"time_vs_depth_{dataset}_{model}_lr{safe_tag(lr)}_link{safe_tag(link_option)}.png",
            dpi=200,
        )
        plt.close()


def plot_rvfl_component_time_vs_depth(summary_df, plot_root):
    out_dir = Path(plot_root) / "rvfl_component_time_vs_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    frame = constant_width_subset(summary_df)
    frame = frame[frame["model"] == "rvfl"] if not frame.empty else frame
    components = [
        "avg_train_forward_time",
        "avg_train_solve_beta_time",
        "avg_backward_time",
        "avg_adam_step_time",
        "avg_projection_time",
    ]
    components = [column for column in components if column in frame.columns]
    if frame.empty or not components:
        return

    for keys, group in frame.groupby(["dataset", "lr", "link_option", "max_width"], dropna=False):
        dataset, lr, link_option, width = keys
        means = group.groupby("depth")[components].mean().sort_index()
        ax = means.plot(kind="bar", stacked=True, figsize=(10, 6))
        ax.set_xlabel("Number of hidden layers")
        ax.set_ylabel("Average train-step component time (seconds)")
        ax.set_title(
            f"RVFL timing vs depth: {dataset}, width={int(width)}, "
            f"lr={lr:g}, link={link_option}"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"components_{dataset}_width{int(width)}_lr{safe_tag(lr)}_link{safe_tag(link_option)}.png",
            dpi=200,
        )
        plt.close()


def plot_accuracy_cost_tradeoff(summary_df, plot_root):
    out_dir = Path(plot_root) / "accuracy_cost_tradeoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    for keys, group in summary_df.groupby(["dataset", "model", "lr", "link_option"], dropna=False):
        dataset, model, lr, link_option = keys
        aggregate = (
            group.groupby(["architecture", "unitary_parameter_count", "total_hidden_units"], dropna=False)["final_test_acc"]
            .mean()
            .reset_index()
        )
        x_column = "unitary_parameter_count" if model == "rvfl" else "total_hidden_units"
        plt.figure(figsize=(9, 6))
        plt.scatter(aggregate[x_column], aggregate["final_test_acc"])
        top = aggregate.nlargest(min(8, len(aggregate)), "final_test_acc")
        for _, row in top.iterrows():
            plt.annotate(row["architecture"], (row[x_column], row["final_test_acc"]), fontsize=8)
        plt.xlabel(x_column.replace("_", " "))
        plt.ylabel("Mean final test accuracy")
        plt.title(f"Accuracy/cost tradeoff: {model} on {dataset}, lr={lr:g}, link={link_option}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            out_dir / f"tradeoff_{dataset}_{model}_lr{safe_tag(lr)}_link{safe_tag(link_option)}.png",
            dpi=200,
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="benchmark_multilayer/results")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    plot_root = result_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    summary_df, history_df = load_all_results(result_root)
    if summary_df.empty:
        print("No summary files found.")
        return

    save_tables(summary_df, history_df, result_root)
    plot_individual_accuracy_curves(history_df, plot_root)
    plot_accuracy_vs_depth(summary_df, plot_root)
    plot_time_vs_depth(summary_df, plot_root)
    plot_rvfl_component_time_vs_depth(summary_df, plot_root)
    plot_accuracy_cost_tradeoff(summary_df, plot_root)
    print(f"Saved combined tables under {result_root}")
    print(f"Saved plots under {plot_root}")


if __name__ == "__main__":
    main()
