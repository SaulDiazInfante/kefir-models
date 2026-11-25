"""Reproduce the fixed structural comparison, extending chains if diagnostics fail."""

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
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(root / "src"),
    )
    cases = []
    for prior, multiplier in [("reference", 1), ("noise_half", 0.5), ("noise_double", 2)]:
        for sharing, kernel in [
            ("shared", "ou"),
            ("independent", "matern32"),
            ("shared", "matern32"),
        ]:
            for cutoff in [108, 120, 132, 168]:
                cases.append((prior, multiplier, sharing, kernel, cutoff))
    for sharing, kernel in [
        ("independent", "ou"),
        ("shared", "ou"),
        ("independent", "matern32"),
        ("shared", "matern32"),
    ]:
        for cutoff in [144, 156]:
            cases.append(("reference", 1, sharing, kernel, cutoff))

    def run(case):
        prior, multiplier, sharing, kernel, cutoff = case
        directory = root / "results/structural" / prior / f"{sharing}_{kernel}" / f"cutoff_{cutoff}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "experiment.json"
        draws = 36000
        for attempt in range(3):
            options = []
            if args.reuse_validated and attempt == 0 and manifest.exists():
                saved = json.loads(manifest.read_text())
                if saved["computation_passed"]:
                    draws = saved["arguments"]["draws"]
                    options = ["--reuse-chains"]
            command = [
                sys.executable,
                "scripts/investigate_structural_discrepancy.py",
                "--kernel",
                kernel,
                "--sharing",
                sharing,
                "--cutoff",
                str(cutoff),
                "--noise-multiplier",
                str(multiplier),
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
                print(f"PASS {prior}/{sharing}_{kernel}/{cutoff} draws={draws}", flush=True)
                return
            draws *= 3
        raise RuntimeError(f"Diagnostics unmet: {directory}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, cases))
    for script in ["summarize_structural_discrepancy.py", "build_structural_tables.py"]:
        subprocess.run([sys.executable, "scripts/" + script], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
