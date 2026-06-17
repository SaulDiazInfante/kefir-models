# Installation

`kefir-models` requires **Python ≥ 3.9** and depends on `pandas`, `numpy`,
`torch`, `torchdiffeq`, and `matplotlib`.

## Get the code

```bash
git clone https://github.com/SaulDiazInfante/kefir-models.git
cd kefir-models
```

## Create an environment

=== "conda"

    ```bash
    conda create -n kefir-models python=3.12 -y
    conda activate kefir-models
    pip install -e ".[dev]"
    ```

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    ```

`torch` and `torchdiffeq` install as prebuilt CPU wheels on most platforms.

## Optional dependency groups

- `".[dev]"` — testing and linting tools (`pytest`, `pytest-cov`, `ruff`, `build`, `pre-commit`).
- `".[docs]"` — documentation tools (`mkdocs`, `mkdocs-material`).

For a runtime-only install use `pip install -e .` or `pip install -r requirements.txt`.

## Verify

```bash
python -c "import kefir_models; print(kefir_models.__version__)"
kefir-ode-fit --help
```
