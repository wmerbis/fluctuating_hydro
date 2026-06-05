# Conservative limiter diagnostics

The conservative finite-volume limiter now stores detailed per-step diagnostics in
`SchellingVoterFVSolver.last_limiter_stats` whenever `limiter_mode="conservative"`.
The existing limiter modes remain unchanged.

## Per-step limiter activity

The diagnostics include the previous activity counters plus direction-resolved
face counts:

- `limited_cells`
- `limited_faces`, `limited_faces_x`, `limited_faces_y`
- `voter_limited_cells`
- `gamma_repair_cells`
- `activations`
- `residual_max_violation`

## Limiter coefficients

The face limiter coefficients are reconstructed from the per-cell budgets after
each conservative step.  The following keys summarize the distribution over all
periodic x- and y-faces:

- `theta_min`, `theta_mean`, `theta_median`, `theta_max`
- `theta_x_min`, `theta_x_mean`, `theta_x_median`, `theta_x_max`
- `theta_y_min`, `theta_y_mean`, `theta_y_median`, `theta_y_max`
- `theta_frac_lt_1`, `theta_frac_lt_0_5`, `theta_frac_lt_0_1`
- `theta_p01`, `theta_p05`, `theta_p10`, `theta_p25`, `theta_p75`, `theta_p90`, `theta_p95`, `theta_p99`

Small `theta_mean`, large low-percentile suppression, or persistent nonzero
`theta_frac_lt_0_5` indicate that the limiter is changing more than rare
overshooting fluxes.

## Flux removal

For each step, the raw face increments are compared with `theta`-limited face
increments.  Each prefix below has `l1_raw`, `l1_limited`, `l2_raw`,
`l2_limited`, and `removed_fraction` fields under the key pattern
`flux_<prefix>_<quantity>`:

- `A`
- `B`
- `occupied`
- `deterministic_A`
- `deterministic_B`
- `deterministic_occupied`
- `schelling_noise_A`
- `schelling_noise_B`
- `schelling_noise_occupied`

The deterministic and Schelling-noise entries apply the same face coefficient to
those reconstructed components.  They should be interpreted as a decomposition of
what the total-flux limiter removed, not as independent limiters.

## Localization

A per-cell limiting activity proxy is computed as the average deficit
`1 - theta` over the four incident periodic faces.  The following Pearson
correlations are reported:

- `limiting_corr_rho_A`
- `limiting_corr_rho_B`
- `limiting_corr_rho_0`
- `limiting_corr_grad_rho`

Strong negative correlation with `rho_0`, positive correlation with density, or
positive correlation with `|grad rho|` helps distinguish vacancy-poor/high-density
limiting from interface-localized limiting.

## Comparison script

Run matched clip/conservative comparisons with identical initial conditions and
RNG streams:

```bash
python scripts/compare_limiter_modes.py --steps 1000 --record-every 100
```

A small instability-threshold scan can be requested with comma-separated beta
values:

```bash
python scripts/compare_limiter_modes.py --beta-scan 6,7,8,9 --csv limiter_scan.csv
```

The script reports mass drift, entropic dissimilarity, order-parameter variance,
Fourier peak amplitude, conservative limiter flux-removal fractions, and
localization correlations.
