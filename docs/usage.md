# Usage

All commands accept `--help` and should be run from the repository root so the
default data and config paths resolve.

## Data

The raw input ships at `data/raw/waterKefirTrialsReferece.csv` (the original file
name spelling is preserved). The CLIs default to
`data/raw/waterKefirTrialsReference.csv` and fall back to the included file
automatically, so they work out of the box.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `kefir-ode-fit` | Fit a Neural ODE to the trial data |
| `kefir-sde-compare` | Compare classical / Neural ODE / Neural SDE fits |
| `kefir-logistic-pinn-compare` | Deterministic and stochastic logistic PINN inverse models |
| `kefir-plot-ode-fit` | Re-plot a saved Neural ODE fit |
| `kefir-plot-pinn-fit` | Re-plot a saved logistic PINN fit |
| `kefir-plot-sde-compare` | Re-plot a saved Neural SDE comparison |
| `kefir-plot-all-models` | Build the combined all-model comparison sequence |

### Examples

```bash
kefir-ode-fit --config configs/ode_fit_config_reference.json
kefir-sde-compare --config configs/sde_compare_config_reference.json
kefir-logistic-pinn-compare --config configs/logistic_pinn_config_reference.json
```

Common flags on the fitting tools: `--data-path`, `--output-dir`, `--epochs`,
`--seed`, `--device {auto,cpu,cuda}`, and `--no-plot`.

!!! note "Where outputs go"
    Generated artifacts (CSVs, `.pt` checkpoints, PNGs) are written to
    directories such as `neural_sde_comparison_outputs/`; these are git-ignored.
    The curated figures used by the report and this site are committed under
    `results/`.

## Python API

```python
from pathlib import Path
from kefir_models import ode_fit

frame = ode_fit.read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv"))
frame, trial_columns = ode_fit.validate_frame(frame)
data = ode_fit.build_training_data(frame, trial_columns)  # normalized tensors + scaling
```

See the [API Reference](api.md) for the full list of public functions.
