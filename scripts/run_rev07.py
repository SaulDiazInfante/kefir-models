"""Reproduce the REV-07 controlled investigation and its aggregate figure."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if not (
        root / "results/neural_sde_comparison_outputs/water_kefir_neural_sde_model.pt"
    ).exists():
        raise FileNotFoundError("Run python scripts/reproduce_report.py before this investigation.")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(root / "src"))

    def run(script, *args):
        subprocess.run(
            [sys.executable, str(root / "scripts" / script), *args], cwd=root, env=env, check=True
        )

    for seed in [7, 17]:
        for cutoff in [168, 108, 120, 132]:
            output = Path("results/rev07")
            if seed != 7:
                output = output / f"seed_{seed}"
            output = output / f"cutoff_{cutoff}"
            run(
                "investigate_rev07.py",
                "--seed",
                str(seed),
                "--cutoff",
                str(cutoff),
                "--output-dir",
                str(output),
            )
    run("diagnose_frozen_transitions.py")
    run(
        "diagnose_frozen_transitions.py",
        "--checkpoint",
        "results/rev07/cutoff_168/compact_moments.pt",
        "--output-dir",
        "results/rev07/frozen_compact_moments",
    )
    run("summarize_rev07.py")


if __name__ == "__main__":
    main()
