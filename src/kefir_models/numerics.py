"""Fixed-grid Itô integration, independent of the requested display times."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class EulerMaruyamaResult:
    samples: Tensor
    steps: int
    clipped_values: int


def euler_maruyama(
    time: np.ndarray,
    initial: Tensor,
    drift: Callable[[Tensor, Tensor], Tensor],
    diffusion: Callable[[Tensor, Tensor], Tensor],
    max_step: float,
    generator: torch.Generator,
    minimum_state: float | None = None,
) -> EulerMaruyamaResult:
    """Sample a fixed-grid Euler-Maruyama approximation at requested times.

    The integration grid depends only on the endpoints and ``max_step``.
    Off-grid outputs linearly interpolate states on this grid; adding output
    times consumes no random numbers and does not alter the integration.
    ``minimum_state`` is an explicit numerical projection, with its frequency
    returned for diagnostics, rather than a positivity guarantee for the SDE.
    """

    times = np.asarray(time, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("time must contain at least two finite output times.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Output times must be strictly increasing.")
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive.")
    steps = math.ceil(float(times[-1] - times[0]) / max_step)
    step = float(times[-1] - times[0]) / steps
    current = initial.clone()
    samples = [current.clone()]
    output_index = 1
    clipped_values = 0
    dt = initial.new_tensor(step)
    with torch.no_grad():
        for index in range(steps):
            left = float(times[0] + index * step)
            right = float(times[0] + (index + 1) * step)
            t = initial.new_tensor(left)
            noise = torch.randn(
                current.shape, device=current.device, dtype=current.dtype, generator=generator
            )
            following = current + drift(t, current) * dt + diffusion(t, current) * dt.sqrt() * noise
            if minimum_state is not None:
                clipped_values += int((following < minimum_state).sum().item())
                following = following.clamp_min(minimum_state)
            if not bool(torch.isfinite(following).all()):
                raise FloatingPointError("Non-finite SDE state; refine the integration step.")
            while output_index < times.size and (
                times[output_index] <= right or index == steps - 1
            ):
                weight = min(max((float(times[output_index]) - left) / step, 0.0), 1.0)
                samples.append(torch.lerp(current, following, weight))
                output_index += 1
            current = following
    return EulerMaruyamaResult(torch.stack(samples).cpu(), steps, clipped_values)
