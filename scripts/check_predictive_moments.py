"""Check the measurement-SD likelihood tail used in the analytic moment proof.

The analytic argument, not this finite grid, establishes divergence. Keeping
calculations on the log scale avoids overflowing a conditional response mean.
No posterior chains are fitted and no independent validation is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.structural_discrepancy import StructuralProblem


def main():
    frame, columns = validate_frame(read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv")))
    rows = []
    for sharing in ["independent", "shared"]:
        for kernel in ["ou", "matern32"]:
            problem = StructuralProblem(
                frame.time.to_numpy() / 12, frame[columns].to_numpy(), kernel, sharing
            )
            for level in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
                z = problem.prior.mean.copy()
                z[-1] = level
                likelihood = problem.log_likelihood(z)
                posterior = problem.log_posterior(z)
                mean, latent, noise = problem.conditional_log_distribution(z, np.array([15.0]))
                log_component_mean = mean[0, 0] + 0.5 * (latent[0, 0] + noise)
                rows.append(
                    {
                        "model": sharing + "_" + kernel,
                        "log_measurement_sd": level,
                        "measurement_sd": np.exp(level),
                        "log_likelihood": likelihood,
                        "log_likelihood_plus_Nz": likelihood + frame[columns].size * level,
                        "log_prior": posterior - likelihood,
                        "log_component_response_mean": log_component_mean,
                        "log_unnormalized_mean_integrand": posterior + log_component_mean,
                    }
                )
    destination = Path("results/prospective/predictive_moment_tail.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(pd.DataFrame(rows).groupby("model").tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
