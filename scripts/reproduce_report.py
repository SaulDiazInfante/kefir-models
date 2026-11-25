"""Reproduce report experiments, tables, and figures without overwriting its PDF."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--table-dir", type=Path, default=Path("docs/generated"))
    parser.add_argument(
        "--reuse-fits",
        action="store_true",
        help="Regenerate tables and combined/site figures from existing fitted outputs.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    def run(module: str, *arguments: str) -> None:
        subprocess.run([sys.executable, "-m", module, *arguments], cwd=root, env=env, check=True)

    neural = args.output_root / "neural_sde_comparison_outputs"
    pinn = args.output_root / "logistic_pinn_outputs"
    if not args.reuse_fits:
        run(
            "kefir_models.sde_compare",
            "--config",
            "configs/sde_compare_config_reference.json",
            "--device",
            "cpu",
            "--sde-paths",
            "1000",
            "--output-dir",
            str(neural),
        )
        run(
            "kefir_models.logistic_pinn_compare",
            "--config",
            "configs/logistic_pinn_config_reference.json",
            "--device",
            "cpu",
            "--output-dir",
            str(pinn),
        )
    run(
        "kefir_models.plot_sde_compare",
        "--input-csv",
        str(neural / "water_kefir_neural_dynamics_comparison.csv"),
        "--output-plot",
        str(neural / "water_kefir_neural_dynamics_comparison_replot.png"),
    )
    run(
        "kefir_models.plot_pinn_fit",
        "--comparison-csv",
        str(pinn / "water_kefir_logistic_pinn_comparison.csv"),
        "--dense-csv",
        str(pinn / "water_kefir_logistic_pinn_dense.csv"),
        "--det-loss-csv",
        str(pinn / "deterministic_logistic_pinn_training_loss.csv"),
        "--sto-loss-csv",
        str(pinn / "stochastic_logistic_pinn_training_loss.csv"),
        "--output-dir",
        str(pinn),
    )
    run(
        "kefir_models.plot_all_models",
        "--neural-comparison-csv",
        str(neural / "water_kefir_neural_dynamics_comparison.csv"),
        "--pinn-comparison-csv",
        str(pinn / "water_kefir_logistic_pinn_comparison.csv"),
        "--pinn-dense-csv",
        str(pinn / "water_kefir_logistic_pinn_dense.csv"),
        "--output-dir",
        str(args.output_root / "all_model_comparison_outputs"),
    )
    run(
        "kefir_models.report_tables",
        "--neural-dir",
        str(neural),
        "--pinn-dir",
        str(pinn),
        "--output-dir",
        str(args.table_dir),
    )
    if (root / args.table_dir).resolve() == root / "docs/generated":
        metrics_path = root / args.table_dir / "model_comparison_metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        page = root / "docs/results.md"
        content = page.read_text(encoding="utf-8")
        start = content.index("| Model | RMSE |")
        end = content.index("## All-model comparison", start)
        table = "| Model | RMSE | R² |\n| --- | --- | --- |\n"
        table += "\n".join(
            f"| {row['model']} | {float(row['rmse']):.4f} | {float(row['r_squared']):.4f} |"
            for row in metrics
        )
        page.write_text(content[:start] + table + "\n\n" + content[end:], encoding="utf-8")
        assets = root / "docs/assets/img"
        assets.mkdir(parents=True, exist_ok=True)
        for source, target in [
            (
                neural / "water_kefir_neural_dynamics_comparison.png",
                "neural_dynamics_comparison.png",
            ),
            (neural / "training_objectives.png", "training_objectives.png"),
            (pinn / "water_kefir_logistic_pinn_comparison.png", "logistic_pinn_comparison.png"),
            (
                args.output_root
                / "all_model_comparison_outputs"
                / "water_kefir_all_models_comparison.png",
                "all_models_comparison.png",
            ),
        ]:
            shutil.copy2(root / source, assets / target)


if __name__ == "__main__":
    main()
