import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, repo_root):
    print("\n" + "=" * 90)
    print(" ".join(str(item) for item in cmd))
    result = subprocess.run(cmd, cwd=repo_root, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def main():
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    python = sys.executable
    result_root = repo_root / "benchmark_multilayer" / "results_smoke"

    if result_root.exists():
        shutil.rmtree(result_root)

    scripts = [
        repo_root / "benchmark_multilayer" / "benchmark_multilayer_job.py",
        repo_root / "benchmark_multilayer" / "make_multilayer_configs.py",
        repo_root / "benchmark_multilayer" / "make_multilayer_plots.py",
    ]
    run([python, "-m", "py_compile", *map(str, scripts)], repo_root)

    common = [
        "--dataset", "mnist",
        "--device", "cpu",
        "--dtype", "float32",
        "--train-limit", "1000",
        "--test-limit", "200",
        "--seed", "0",
        "--result-root", str(result_root),
    ]

    jobs = [
        ["--model", "rvfl", "--widths", "20,15", "--link-option", "multi", "--rvfl-epochs", "2"],
        ["--model", "rvfl", "--widths", "12,12,12,12,12", "--link-option", "direct", "--rvfl-epochs", "1"],
        ["--model", "standard_nn", "--widths", "20,15", "--nn-epochs", "2"],
    ]

    for job in jobs:
        run(
            [
                python,
                "benchmark_multilayer/benchmark_multilayer_job.py",
                *common,
                *job,
                "--lr", "0.003",
                "--lamb", "0.01",
                "--scalings", "0",
                "--ortho-method", "qr",
            ],
            repo_root,
        )

    run(
        [
            python,
            "benchmark_multilayer/make_multilayer_plots.py",
            "--result-root",
            str(result_root),
        ],
        repo_root,
    )

    expected = [
        result_root / "summary_all.csv",
        result_root / "history_all.csv",
        result_root / "architecture_rankings.csv",
        result_root / "architecture_aggregate.csv",
        result_root / "plots" / "accuracy_curves",
        result_root / "plots" / "accuracy_vs_depth",
        result_root / "plots" / "time_vs_depth",
    ]
    for path in expected:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Found: {path}")

    print("\nMULTILAYER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
