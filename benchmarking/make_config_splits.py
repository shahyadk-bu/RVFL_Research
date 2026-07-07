from pathlib import Path


def main():
    out_dir = Path("benchmarking")
    out_dir.mkdir(parents=True, exist_ok=True)

    lamb = 1e-2
    lamb_tag = "lam_1e-2"

    datasets_to_widths = {
        "mnist": [100, 200, 500, 1000, 2000, 3000, 4000, 5000],
        "fashionmnist": [100, 200, 500, 1000, 2000, 3000, 4000, 5000],
        "cifar10": [100, 200, 500, 1000, 2000],
    }

    # Same LR sweep for both standard NN and RVFL.
    lr_options = [
        (1e-3, "lr_1e-3"),
        (2e-3, "lr_2e-3"),
        (3e-3, "lr_2e-3"),
        (1e-2, "lr_2e-3"),
        (2e-2, "lr_2e-3"),
        (3e-2, "lr_2e-3"),
    ]

    buckets = {
        "nn": [],
        "rvfl_medium": [],
        "rvfl_heavy": [],
    }

    for lr, lr_tag in lr_options:
        result_root = f"benchmark_results/{lamb_tag}/{lr_tag}"

        for dataset, widths in datasets_to_widths.items():
            for width in widths:
                # ----------------------------------------------------
                # Standard NN job.
                # ----------------------------------------------------
                nn_line = (
                    f"--dataset {dataset} "
                    f"--model standard_nn "
                    f"--width {width} "
                    f"--lr {lr:g} "
                    f"--result-root {result_root}"
                )

                buckets["nn"].append(nn_line)

                # ----------------------------------------------------
                # RVFL job.
                # ----------------------------------------------------
                rvfl_line = (
                    f"--dataset {dataset} "
                    f"--model rvfl "
                    f"--width {width} "
                    f"--lr {lr:g} "
                    f"--lamb {lamb:g} "
                    f"--result-root {result_root}"
                )

                if width <= 2000:
                    buckets["rvfl_medium"].append(rvfl_line)
                else:
                    buckets["rvfl_heavy"].append(rvfl_line)

    for name, lines in buckets.items():
        path = out_dir / f"configs_{name}.txt"

        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")

        print(f"Wrote {len(lines)} jobs to {path}")

    print("\nExpected counts:")
    print("  configs_nn.txt:          147 jobs")
    print("  configs_rvfl_medium.txt: 105 jobs")
    print("  configs_rvfl_heavy.txt:   42 jobs")
    print("  total:                   294 jobs")


if __name__ == "__main__":
    main()