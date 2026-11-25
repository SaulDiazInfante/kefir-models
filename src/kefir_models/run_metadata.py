"""Persist the effective settings and inputs behind a numerical experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchdiffeq


def save_run_metadata(
    args: argparse.Namespace,
    data_path: Path,
    device: torch.device,
    diagnostics: dict,
) -> None:
    """Record versions, content hashes, and actual CLI values, including overrides."""

    source_dir = Path(__file__).parent
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_dir.glob("*.py"))
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    metadata = {
        "arguments": vars(args),
        "device": str(device),
        "stochastic_interpretation": "Ito",
        "transition_likelihood": "one_observation_step_Euler_Maruyama_Gaussian",
        "simulation": "fixed_grid_Euler_Maruyama_with_linear_output_interpolation",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torchdiffeq": torchdiffeq.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "diagnostics": diagnostics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
