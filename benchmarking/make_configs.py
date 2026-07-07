from pathlib import Path


def main():
    config_path = Path("benchmarking/configs.txt")

    datasets_to_widths = {
        "mnist": [100, 200, 500, 1000, 2000, 3000, 4000, 5000],
        "fashionmnist": [100, 200, 500, 1000, 2000, 3000, 4000, 5000],
        "cifar10": [100, 200, 500, 1000, 2000],
    }

    models = ["rvfl", "standard_nn"]

    lr_options = [
        (3e-3, "lr_3e-3"),
        (2e-3, "lr_2e-3"),
        (1e-2, "lr_1e-2"),
    ]

    lines = []

    for lr, lr_tag in lr_options:
        for dataset, widths in datasets_to_widths.items():
            for model in models:
                for width in widths:
                    result_root = f"benchmark_results/lam_1e-2/{lr_tag}"

                    line = (
                        f"--dataset {dataset} "
                        f"--model {model} "
                        f"--width {width} "
                        f"--lr {lr:g} "
                        f"--result-root {result_root}"
                    )

                    lines.append(line)

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} jobs to {config_path}")
    print(f"Use this number in the qsub array line: #$ -t 1-{len(lines)}")


if __name__ == "__main__":
    main()