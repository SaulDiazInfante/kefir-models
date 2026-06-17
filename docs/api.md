# API Reference

A curated overview of the public API. Every module also exposes a `main()`
entry point used by the corresponding console script.

## `kefir_models.ode_fit`

Neural ODE fitting and the shared data pipeline.

- `Scaling` — frozen dataclass of affine scaling constants (`time_min`, `time_scale`, `value_mean`, `value_std`).
- `TrainingData` — frozen dataclass bundling normalized tensors, mask, scaling, trial columns, and the raw frame.
- `NeuralODEFunction(hidden_units, hidden_layers)` — `nn.Module` neural right-hand side `dy/dt = f_theta(t, y)`.
- `read_trial_csv(path)` — robust CSV reader that trims malformed rows.
- `resolve_data_path(path)` — resolve the requested path, falling back to the bundled file name.
- `validate_frame(frame)` — validate/sort the data; returns `(frame, trial_columns)`.
- `build_training_data(frame, trial_columns)` — produce normalized training tensors as `TrainingData`.
- `masked_mse(prediction, target, mask)` — MSE over observed entries only.
- `population_initial_state(target)` — single normalized initial state.
- `solve_trajectory(function, initial_state, time_points, method)` — integrate the Neural ODE.
- `train_model(...)` — train and return `(model, fitted_curve, loss_history)`.
- `inverse_transform_values(values, scaling)` — map normalized values to the original scale.
- `build_output_frame(data, prediction)` / `compute_rmse(frame, trial_columns)` / `save_artifacts(...)`.
- `select_device(device_name)` — resolve `"auto" | "cpu" | "cuda"`.

## `kefir_models.sde_compare`

Classical / Neural ODE / Neural SDE comparison and metrics.

- `NeuralSDEFunction(hidden_units, hidden_layers, min_diffusion)` — drift and diffusion networks; `.drift(t, y)`, `.diffusion(t, y)`.
- `build_mlp(hidden_units, hidden_layers)` — small fully connected network.
- `TransitionBatch`, `SDEPrediction`, `ClassicalLogisticFit` — result dataclasses.
- `build_transition_batch(data, device)` / `transition_negative_log_likelihood(model, batch)`.
- `train_sde_model(...)` / `simulate_sde_paths(...)`.
- `inverse_softplus(value)` / `logistic_growth_prediction(time, initial, r, K)`.
- `train_classical_logistic_model(...)`.
- Metrics: `compute_rmse`, `compute_r_squared`, `observed_fitted_pairs`, `gaussian_information_criteria`, `count_trainable_parameters`, `compute_interval_coverage`.
- Frames/plots: `build_comparison_frame`, `build_metrics_frame`, `build_training_objective_scales_frame`, `save_comparison_plot`, `save_training_objective_scale_plot`, `save_outputs`.

## `kefir_models.logistic_pinn_compare`

Deterministic and stochastic logistic PINN inverse models.

- `parse_args(args=None)` — command-line parser.
- `main()` — fit the deterministic and stochastic logistic PINNs and write comparison outputs.

See the module source for the trajectory-network architecture and loss terms.

## `kefir_models.plot_style`

Shared plotting style used by every figure.

- Constants: `MODEL_COLORS`, `MODEL_LINE_STYLES`, `MODEL_LABELS`, `TRIAL_COLORS`, `TRIAL_MARKERS`, figure-size/layout settings.
- `model_line_style(model_key, **overrides)` — copy of a model's line style.
- `trial_line_style(index, **overrides)` — cycling observed-trial marker style.
- `deduplicated_legend(axis, **overrides)` / `apply_comparison_axis_style(axis)`.
- `save_fixed_layout_png(fig, path)` / `save_tight_png(fig, path)`.

## Plotting CLIs

`kefir_models.plot_ode_fit`, `kefir_models.plot_pinn_fit`,
`kefir_models.plot_sde_compare`, and `kefir_models.plot_all_models` each expose
`parse_args(args=None)` and `main()` for rebuilding figures from saved CSV
outputs. `plot_all_models` additionally provides `infer_trial_columns(frame)`.
