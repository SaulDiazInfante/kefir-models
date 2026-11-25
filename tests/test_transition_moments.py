import math

import pytest
import torch
from torch import nn

from kefir_models.sde_compare import (
    NeuralSDEFunction,
    TransitionBatch,
    transition_negative_log_likelihood,
)
from kefir_models.transition_moments import gaussian_moment_nll, gaussian_transition_moments


def batch(dtype=torch.float64):
    return TransitionBatch(
        torch.zeros((1, 1), dtype=dtype),
        torch.ones((1, 1), dtype=dtype),
        torch.ones((1, 1), dtype=dtype),
        torch.ones((1, 1), dtype=dtype),
    )


def test_one_step_matches_original_objective_and_gradients():
    torch.manual_seed(7)
    model = NeuralSDEFunction(4, 1, 0.001).double()
    original = transition_negative_log_likelihood(model, batch())
    refined = gaussian_moment_nll(model, batch(), 1)
    torch.testing.assert_close(original, refined)
    original_grad = torch.autograd.grad(original, tuple(model.parameters()))
    refined_grad = torch.autograd.grad(refined, tuple(model.parameters()))
    for first, second in zip(original_grad, refined_grad):
        torch.testing.assert_close(first, second)


class OU(nn.Module):
    def drift(self, time, state):
        return -1.2 * state

    def diffusion(self, time, state):
        return torch.full_like(state, 0.4)


def test_ou_moments_converge_to_known_gaussian_transition():
    exact_mean = math.exp(-1.2)
    exact_variance = 0.4**2 * (1 - math.exp(-2.4)) / 2.4
    coarse_mean, coarse_var = gaussian_transition_moments(OU(), batch(), 16)
    fine_mean, fine_var = gaussian_transition_moments(OU(), batch(), 256)
    assert float(abs(fine_mean - exact_mean)) < float(abs(coarse_mean - exact_mean)) / 10
    assert float(fine_var - 1e-6) == pytest.approx(exact_variance, rel=0.004)


def test_gbm_moments_converge_despite_non_gaussian_distribution():
    class GBM(nn.Module):
        def drift(self, time, state):
            return 0.2 * state

        def diffusion(self, time, state):
            return 0.3 * state

    mean, var = gaussian_transition_moments(GBM(), batch(), 256)
    assert float(mean) == pytest.approx(math.exp(0.2), rel=0.001)
    assert float(var - 1e-6) == pytest.approx(math.exp(0.4) * math.expm1(0.09), rel=0.004)


def test_multistep_objective_has_finite_parameter_gradients():
    torch.manual_seed(7)
    model = NeuralSDEFunction(4, 1, 0.001).double()
    loss = gaussian_moment_nll(model, batch(), 16)
    for grad in torch.autograd.grad(loss, tuple(model.parameters())):
        assert bool(torch.isfinite(grad).all())


def test_invalid_substeps_rejected():
    with pytest.raises(ValueError):
        gaussian_transition_moments(OU(), batch(), 0)
