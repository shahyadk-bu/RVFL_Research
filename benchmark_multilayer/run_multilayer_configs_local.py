import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def read_configs(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def main():
    parser = argparse.ArgumentParser(
        description="Run a multilayer benchmark config file locally, one job at a time."
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--rvfl-epochs", type=int, default=None)
    parser.add_argument("--nn-epochs", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    benchmark_script = repo_root / "benchmark_multilayer" / "benchmark_multilayer_job.py"
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")
    if args.start_index < 1:
        raise ValueError("--start-index is 1-based and must be at least 1.")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("--max-jobs must be positive.")

    configs = read_configs(config_path)
    selected = configs[args.start_index - 1 :]
    if args.max_jobs is not None:
        selected = selected[: args.max_jobs]

    if not selected:
        raise ValueError("No configurations were selected.")

    failures = []
    total = len(selected)

    for local_number, config in enumerate(selected, start=1):
        source_index = args.start_index + local_number - 1
        cmd = [
            sys.executable,
            str(benchmark_script),
            *shlex.split(config, posix=(sys.platform != "win32")),
            "--device", args.device,
            "--dtype", args.dtype,
        ]

        if args.rvfl_epochs is not None:
            cmd.extend(["--rvfl-epochs", str(args.rvfl_epochs)])
        if args.nn_epochs is not None:
            cmd.extend(["--nn-epochs", str(args.nn_epochs)])

        print("\n" + "=" * 100)
        print(f"Local job {local_number}/{total} | config line {source_index}")
        print(config)
        print("=" * 100)

        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            failures.append((source_index, result.returncode, config))
            if not args.continue_on_error:
                break

    print("\n" + "=" * 100)
    if failures:
        print(f"Finished with {len(failures)} failure(s):")
        for index, returncode, config in failures:
            print(f"  line {index}, return code {returncode}: {config}")
        raise SystemExit(1)

    print(f"Successfully completed {total} local job(s).")


if __name__ == "__main__":
    main()
