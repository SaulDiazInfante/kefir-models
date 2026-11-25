import numpy as np
import pytest
import torch

from kefir_models.numerics import euler_maruyama


def test_ito_geometric_brownian_mean_and_variance():
    # dX = b X dW has mean X0, unlike its uncorrected Stratonovich version.
    result = euler_maruyama(
        np.array([0.0, 1.0]),
        torch.ones((30000, 1)),
        drift=lambda t, x: torch.zeros_like(x),
        diffusion=lambda t, x: 0.6 * x,
        max_step=0.01,
        generator=torch.Generator().manual_seed(4),
    )
    final = result.samples[-1, :, 0]
    assert float(final.mean()) == pytest.approx(1.0, abs=0.015)
    assert float(final.var(unbiased=False)) == pytest.approx(np.exp(0.36) - 1, abs=0.03)


def test_projection_is_counted_explicitly():
    result = euler_maruyama(
        np.array([0.0, 1.0]),
        torch.ones((2, 1)),
        drift=lambda t, x: -2 * x,
        diffusion=lambda t, x: torch.zeros_like(x),
        max_step=1.0,
        generator=torch.Generator().manual_seed(1),
        minimum_state=1e-6,
    )
    assert result.clipped_values == 2
    assert bool((result.samples[-1] == 1e-6).all())


@pytest.mark.parametrize(
    "times,step", [([0.0, 0.0], 0.1), ([0.0, np.inf], 0.1), ([0.0, 1.0], 0.0), ([0.0, 1.0], np.nan)]
)
def test_invalid_time_or_step_is_rejected(times, step):
    with pytest.raises(ValueError):
        euler_maruyama(
            np.array(times), torch.ones(1), lambda t, x: x, lambda t, x: x, step, torch.Generator()
        )
