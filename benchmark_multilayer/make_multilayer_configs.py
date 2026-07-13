import argparse
from pathlib import Path


DATASET_INPUT_DIMS = {
    "mnist": 784,
    "fashionmnist": 784,
    "cifar10": 3072,
}


def parse_csv(text, cast=str):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def architecture_tag(widths):
    return "x".join(str(width) for width in widths)


def unique_architectures(items):
    seen = set()
    output = []
    for family, widths in items:
        key = tuple(widths)
        if key not in seen:
            seen.add(key)
            output.append((family, widths))
    return output


def build_architectures(mode):
    architectures = []

    if mode == "pilot":
        for width in [50, 100, 200]:
            for depth in range(1, 6):
                architectures.append(("constant_width", [width] * depth))

        architectures.extend(
            [
                ("expanding", [100, 200]),
                ("funnel", [200, 100]),
                ("bottleneck", [200, 50, 200]),
                ("expanding", [50, 100, 200]),
                ("funnel", [200, 100, 50]),
            ]
        )

    elif mode == "full":
        # Equal-width families isolate the effect of depth most cleanly.
        for width in [100, 200, 500]:
            for depth in range(1, 6):
                architectures.append(("constant_width", [width] * depth))

        # A deliberately limited shape sweep. Avoid a full Cartesian product.
        architectures.extend(
            [
                ("expanding", [100, 200]),
                ("funnel", [200, 100]),
                ("expanding", [200, 500]),
                ("funnel", [500, 200]),
                ("expanding", [100, 200, 500]),
                ("funnel", [500, 200, 100]),
                ("bottleneck", [500, 100, 500]),
                ("bottleneck", [200, 500, 200]),
                ("expanding", [100, 200, 200, 500]),
                ("funnel", [500, 200, 200, 100]),
                ("bottleneck", [500, 200, 100, 500]),
                ("expanding", [100, 150, 200, 300, 500]),
                ("funnel", [500, 300, 200, 150, 100]),
                ("bottleneck", [500, 250, 100, 250, 500]),
            ]
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return unique_architectures(architectures)


def estimate_design_dim(dataset, widths, link_option):
    input_dim = DATASET_INPUT_DIMS[dataset]
    if link_option == "multi":
        return input_dim + sum(widths)
    if link_option == "direct":
        return input_dim + widths[-1]
    return widths[-1]


def rvfl_bucket(dataset, widths, link_option):
    # QR work is roughly cubic in each layer width; the ridge solve depends on p.
    qr_proxy = sum(width ** 3 for width in widths)
    design_dim = estimate_design_dim(dataset, widths, link_option)

    if qr_proxy <= 5 * (200 ** 3) and design_dim <= 2500:
        return "rvfl_light"
    if qr_proxy <= 5 * (500 ** 3) and design_dim <= 5000:
        return "rvfl_medium"
    return "rvfl_heavy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--datasets", default="mnist,fashionmnist,cifar10")
    parser.add_argument("--models", default="rvfl")
    parser.add_argument("--lrs", default="0.003")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--link-options", default="multi")
    parser.add_argument("--lamb", type=float, default=1e-2)
    parser.add_argument("--scaling", type=float, default=0.0)
    parser.add_argument("--result-root", default="benchmark_multilayer/results")
    parser.add_argument("--output-dir", default="benchmark_multilayer/configs")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    models = parse_csv(args.models)
    lrs = parse_csv(args.lrs, float)
    seeds = parse_csv(args.seeds, int)
    link_options = parse_csv(args.link_options)

    valid_datasets = set(DATASET_INPUT_DIMS)
    valid_models = {"rvfl", "standard_nn"}
    valid_links = {"none", "direct", "multi"}

    if not set(datasets) <= valid_datasets:
        raise ValueError(f"Invalid datasets: {set(datasets) - valid_datasets}")
    if not set(models) <= valid_models:
        raise ValueError(f"Invalid models: {set(models) - valid_models}")
    if not set(link_options) <= valid_links:
        raise ValueError(f"Invalid link options: {set(link_options) - valid_links}")

    architectures = build_architectures(args.mode)
    buckets = {
        "rvfl_light": [],
        "rvfl_medium": [],
        "rvfl_heavy": [],
        "nn": [],
    }

    for dataset in datasets:
        for model in models:
            for lr in lrs:
                for seed in seeds:
                    for family, widths in architectures:
                        widths_arg = ",".join(str(width) for width in widths)

                        if model == "standard_nn":
                            line = (
                                f"--dataset {dataset} --model standard_nn "
                                f"--widths {widths_arg} --architecture-family {family} "
                                f"--lr {lr:g} --seed {seed} --result-root {args.result_root}"
                            )
                            buckets["nn"].append(line)
                            continue

                        for link_option in link_options:
                            line = (
                                f"--dataset {dataset} --model rvfl "
                                f"--widths {widths_arg} --architecture-family {family} "
                                f"--link-option {link_option} --scalings {args.scaling:g} "
                                f"--lr {lr:g} --lamb {args.lamb:g} --seed {seed} "
                                f"--result-root {args.result_root}"
                            )
                            buckets[rvfl_bucket(dataset, widths, link_option)].append(line)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Architectures: {len(architectures)}")
    print("Architecture list:")
    for family, widths in architectures:
        print(f"  {family:15s} {architecture_tag(widths)}")

    for bucket, lines in buckets.items():
        path = output_dir / f"configs_{args.mode}_{bucket}.txt"
        with path.open("w") as handle:
            for line in lines:
                handle.write(line + "\n")
        print(f"Wrote {len(lines):4d} jobs to {path}")

    total = sum(len(lines) for lines in buckets.values())
    print(f"Total jobs: {total}")
    print("Submit each nonempty file with:")
    print(
        "  qsub -t 1-$(wc -l < CONFIG_FILE) -tc 4 "
        "-v CONFIG_FILE=CONFIG_FILE benchmark_multilayer/run_multilayer_array.qsub"
    )


if __name__ == "__main__":
    main()
