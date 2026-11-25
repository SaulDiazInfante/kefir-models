"""Reproduce Bayesian observation-model fits, diagnostics, and report artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-validated",
        action="store_true",
        help="Reuse existing posterior draws that already met computation criteria.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(root / "src"),
    )
    for folder, multiplier, kinds in [
        ("reference", 1.0, ["iid", "ou"]),
        ("noise_half", 0.5, ["ou"]),
        ("noise_double", 2.0, ["ou"]),
    ]:
        for kind in kinds:
            for cutoff in [168, 108, 120, 132]:
                directory = root / "results/bayesian" / folder / kind / f"cutoff_{cutoff}"
                draws = 12000 if kind == "iid" else 36000
                if folder == "reference" and kind == "ou" and cutoff == 168:
                    draws = 12000
                if folder == "noise_half" and cutoff in [120, 132]:
                    draws = 108000
                for attempt in range(3):
                    options = []
                    manifest = directory / "experiment.json"
                    if args.reuse_validated and attempt == 0 and manifest.exists():
                        saved = json.loads(manifest.read_text())
                        if saved["computation_passed"]:
                            draws = saved["arguments"]["draws"]
                            options = ["--reuse-chains"]
                    command = [
                        sys.executable,
                        str(root / "scripts/investigate_bayesian_observation.py"),
                        "--kind",
                        kind,
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
                    subprocess.run(command, cwd=root, env=env, check=True)
                    if json.loads(manifest.read_text())["computation_passed"]:
                        break
                    draws *= 3
                else:
                    raise RuntimeError(
                        f"MCMC criteria remain unmet for {directory}; inspect diagnostics."
                    )
    for script in ["summarize_bayesian_observation.py", "build_iteration_tables.py"]:
        subprocess.run(
            [sys.executable, str(root / "scripts" / script)], cwd=root, env=env, check=True
        )


if __name__ == "__main__":
    main()
