"""Reproduce the fixed two-strength tempered-noise comparison."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-validated", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    models = [
        ("independent", "ou"),
        ("shared", "ou"),
        ("independent", "matern32"),
        ("shared", "matern32"),
    ]
    cases = [
        (eta, sharing, kernel, cutoff)
        for eta in [2, 8]
        for sharing, kernel in models
        for cutoff in ([108, 120, 132, 144, 156, 168] if eta == 2 else [108, 120, 132, 168])
    ]

    def run(case):
        eta, sharing, kernel, cutoff = case
        directory = (
            root / "results/tempered" / f"eta_{eta}" / f"{sharing}_{kernel}" / f"cutoff_{cutoff}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "experiment.json"
        draws = 36000
        for attempt in range(3):
            options = []
            if args.reuse_validated and attempt == 0 and manifest.exists():
                old = json.loads(manifest.read_text())
                if old["computation_passed"]:
                    draws = old["arguments"]["draws"]
                    options = ["--reuse-chains"]
            command = [
                sys.executable,
                "scripts/investigate_tempered_noise.py",
                "--eta",
                str(eta),
                "--kernel",
                kernel,
                "--sharing",
                sharing,
                "--cutoff",
                str(cutoff),
                "--draws",
                str(draws),
                "--output-dir",
                str(directory),
                *options,
            ]
            with (directory / "run.log").open("w") as log:
                subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if json.loads(manifest.read_text())["computation_passed"]:
                print(
                    f"PASS eta={eta} {sharing}_{kernel} cutoff={cutoff} draws={draws}", flush=True
                )
                return
            draws *= 3
        raise RuntimeError(f"Computation gates remain unmet: {directory}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, cases))
    for script in ["summarize_tempered_noise.py", "build_tempered_tables.py"]:
        subprocess.run([sys.executable, "scripts/" + script], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
