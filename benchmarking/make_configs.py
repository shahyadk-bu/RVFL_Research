from pathlib import Path


datasets_to_widths = {
    "mnist": [100, 200, 500, 1000, 2000, 3000, 4000, 5000],
    "fashionmnist": [100, 200, 500, 1000, 2000, 3000, 4000, 5000],
    "cifar10": [100, 200, 500, 1000, 2000],
}

models = ["rvfl", "standard_nn"]

out_path = Path("benchmarking/configs.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

num_jobs = 0

with open(out_path, "w") as f:
    for dataset, widths in datasets_to_widths.items():
        for width in widths:
            for model in models:
                f.write(f"--dataset {dataset} --model {model} --width {width}\n")
                num_jobs += 1

print(f"Wrote {num_jobs} jobs to {out_path}")
print("Use this number in the Slurm array line.")