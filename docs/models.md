# Models

All models are fitted to three water kefir trials measured at 15 time points
(45 observations). The Neural ODE and the logistic PINNs are treated as
*population* models: one fitted curve is compared against every trial.

## Data preprocessing

Time is min–max normalized to `tau in [0, 1]`. The Neural ODE and Neural SDE
train on a standardized response `z = (y - mean) / std`; diagnostics are always
reported back on the original response scale. The logistic PINNs work directly
on the original response scale (losses divided by the response variance).

## Classical logistic ODE

The Verhulst baseline `dy/dtau = r * y * (1 - y / K)`, fitted via its exact
closed-form solution. The unknowns `r > 0` and `K > max(y0)` are obtained from
unconstrained variables through a `softplus` reparameterization. Only two
physical parameters are fitted.

## Neural ODE

A non-autonomous neural right-hand side `dy/dt = f_theta(t, y)` (a small MLP with
`tanh` activations), integrated with `torchdiffeq` (`dopri5` or `rk4`). One
population trajectory is fitted against all trials with a masked MSE loss.

## Neural SDE

An Itô SDE `dY = mu_theta(t, Y) dt + sigma_theta(t, Y) dW`, where both drift and
diffusion are neural networks. It is trained with an Euler–Maruyama Gaussian
transition likelihood and then simulated forward (Monte Carlo) to estimate the
predictive mean and central predictive intervals.

## Deterministic logistic PINN

A physics-informed neural network whose trajectory network is constrained by the
logistic ODE residual. It identifies the physical parameters (`eta`, `r`, `K`)
while balancing data, initial-condition, and collocation (physics) losses.

## Stochastic logistic PINN / SDE

Extends the deterministic PINN with a multiplicative diffusion parameter
`sigma`, identified through an Euler–Maruyama transition likelihood, yielding a
predictive band in addition to the drift trajectory.

## Scoring

Models are compared with RMSE, R², AIC, and BIC computed from observed residuals
on the original scale. The stochastic models additionally report predictive
interval coverage and mean interval width. See [Results](results.md).
