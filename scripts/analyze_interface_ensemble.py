#!/usr/bin/env python3
"""Analyze interface observables from ``interface_ensemble.py`` outputs.

The script reads the individual ``run_*.npz`` files, recomputes ensemble
statistics across realizations, writes a tidy CSV, and produces comparison
plots with logarithmic time axes.

Typical use (run from the repository root)
------------------------------------------
python scripts/analyze_interface_ensemble.py \
    --input interface_ensemble \
    --conditions sv_stoc schelling_stoc

By default the shaded error bands are 95% confidence intervals for the
ensemble mean.  Use ``--error sem`` for +/- 1 standard error or ``--error std``
for +/- 1 ensemble standard deviation.

Notes
-----
* Statistics are computed from per-run observables, not from the spatially
  ensemble-averaged density field.  This matters because averaging the fields
  first can wash out stochastic interfaces.
* t=0 is omitted from figures with a logarithmic time axis, because log(0) is
  undefined.  The t=0 values remain present in the exported CSV.
* The effective logarithmic slope b_eff(t) = dI/dln(t) is estimated by fitting
  each individual run locally in log time, then averaging the run-level slopes.
  Only centered windows are used, so no boundary-extrapolated slopes appear.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.stats import t as student_t
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - fitting requires scipy
    student_t = None
    curve_fit = None


# Human-readable labels used in figures.
CONDITION_LABELS = {
    "sv_stoc": "Schelling + Voter",
    "schelling_stoc": "Schelling only",
    "sv_det": "Schelling + Voter (det.)",
    "schelling_det": "Schelling only (det.)",
}

# Keys that are metadata rather than frame-by-frame observables.
NON_METRIC_KEYS = {
    "condition",
    "run_id",
    "seed",
    "time",
    "snapshot_indices",
    "snapshot_times",
    "snapshots",
}


# -----------------------------------------------------------------------------
# Loading and statistics
# -----------------------------------------------------------------------------
def _discover_run_files(condition_dir: Path) -> List[Path]:
    files = sorted(condition_dir.glob("run_*.npz"))
    if not files:
        raise FileNotFoundError(f"No run_*.npz files found in {condition_dir}")
    return files


def load_condition(condition_dir: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Path]]:
    """Load all per-run time series for one condition.

    Returns
    -------
    time : (nt,) array
    metrics : dict[str, (nruns, nt) array]
    files : list of files actually included
    """
    files = _discover_run_files(condition_dir)

    time = None
    per_metric: Dict[str, List[np.ndarray]] = {}
    common_metrics = None

    # First pass: determine metrics common to all run files.
    file_metric_sets = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            metric_set = {
                k for k in z.files
                if k not in NON_METRIC_KEYS and np.asarray(z[k]).ndim == 1
            }
            file_metric_sets.append(metric_set)

    common_metrics = set.intersection(*file_metric_sets)
    if not common_metrics:
        raise RuntimeError(f"No common time-series observables found in {condition_dir}")

    for name in sorted(common_metrics):
        per_metric[name] = []

    for path in files:
        with np.load(path, allow_pickle=False) as z:
            this_time = np.asarray(z["time"], dtype=float)
            if time is None:
                time = this_time.copy()
            elif this_time.shape != time.shape or not np.allclose(this_time, time):
                raise RuntimeError(f"Time grid in {path} differs from the first run")

            for name in per_metric:
                arr = np.asarray(z[name], dtype=float)
                if arr.shape != time.shape:
                    raise RuntimeError(
                        f"Observable {name!r} in {path} has shape {arr.shape}; "
                        f"expected {time.shape}"
                    )
                per_metric[name].append(arr)

    metrics = {name: np.stack(arrays, axis=0) for name, arrays in per_metric.items()}

    # Add normalized versions for *every* geometric interface scale.  The
    # original simulation script only stored sigma=1 explicitly, but these
    # per-run normalizations are useful for comparing scale dependence.
    for name in sorted(list(metrics)):
        if name.startswith("interface_density_sigma"):
            suffix = name.removeprefix("interface_density_sigma")
            x = metrics[name]
            denom = x[:, [0]]
            denom = np.where(np.abs(denom) > 1e-15, denom, np.nan)
            metrics[f"interface_ratio_sigma{suffix}"] = x / denom

    if "voter_interface_density" in metrics and "voter_interface_ratio" not in metrics:
        x = metrics["voter_interface_density"]
        denom = np.where(np.abs(x[:, [0]]) > 1e-15, x[:, [0]], np.nan)
        metrics["voter_interface_ratio"] = x / denom

    return time, metrics, files


def ensemble_stats(x: np.ndarray, error_mode: str = "ci95") -> Dict[str, np.ndarray]:
    """Compute mean, std, SEM and selected uncertainty half-width over runs."""
    n = np.sum(np.isfinite(x), axis=0)
    mean = np.nanmean(x, axis=0)

    # ddof=1 is undefined for n=1; handle that explicitly.
    std = np.zeros_like(mean)
    valid_many = n > 1
    if np.any(valid_many):
        # np.nanstd with ddof=1 may warn for positions with one sample; mask later.
        with np.errstate(invalid="ignore", divide="ignore"):
            std_all = np.nanstd(x, axis=0, ddof=1)
        std[valid_many] = std_all[valid_many]
        std[~valid_many] = 0.0

    sem = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)

    if error_mode == "std":
        err = std
    elif error_mode == "sem":
        err = sem
    elif error_mode == "ci95":
        if student_t is None:
            factor = np.full_like(mean, 1.96, dtype=float)
        else:
            # Student-t confidence interval; tends to 1.96 for large n.
            dof = np.maximum(n - 1, 1)
            factor = student_t.ppf(0.975, dof)
            factor = np.where(n > 1, factor, 0.0)
        err = factor * sem
    else:
        raise ValueError(f"Unknown error mode: {error_mode}")

    return {
        "n": n.astype(float),
        "mean": mean,
        "std": std,
        "sem": sem,
        "err": err,
    }


def geometric_interface_names(metrics: Dict[str, np.ndarray]) -> List[str]:
    names = [k for k in metrics if k.startswith("interface_density_sigma")]

    def sigma_value(name: str) -> float:
        return float(name.removeprefix("interface_density_sigma"))

    return sorted(names, key=sigma_value)


# -----------------------------------------------------------------------------
# Scaling fits
# -----------------------------------------------------------------------------
def _parse_window(spec: str) -> Tuple[float, float]:
    """Parse a command-line fit window of the form 'tmin:tmax'."""
    try:
        lo, hi = spec.split(":", 1)
        lo = float(lo)
        hi = float(hi)
    except Exception as exc:
        raise ValueError(f"Invalid fit window {spec!r}; expected TMIN:TMAX") from exc
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo:
        raise ValueError(f"Invalid fit window {spec!r}; require 0 < TMIN < TMAX")
    return lo, hi


def _log_model(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.log(t)


def _power_model(t: np.ndarray, a: float, c: float, alpha: float) -> np.ndarray:
    return a + c * np.power(t, alpha)


def _safe_sigma(sem: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Make strictly positive fit uncertainties while preserving measured SEMs.

    The ensemble SEM can be exactly zero at the initial/absorbing state or for a
    deterministic control.  curve_fit cannot use zero sigma.  We replace only
    such values by a small data-scale floor; finite nonzero SEMs are untouched.
    """
    sem = np.asarray(sem, dtype=float).copy()
    finite_pos = sem[np.isfinite(sem) & (sem > 0)]
    yscale = np.nanstd(y)
    if not np.isfinite(yscale) or yscale <= 0:
        yscale = max(np.nanmax(np.abs(y)), 1.0) if np.any(np.isfinite(y)) else 1.0
    if finite_pos.size:
        floor = max(float(np.nanmedian(finite_pos)) * 1.0e-3, yscale * 1.0e-8, 1.0e-12)
    else:
        floor = max(yscale * 1.0e-6, 1.0e-12)
    sem[~np.isfinite(sem) | (sem <= 0)] = floor
    return sem


def _goodness(y: np.ndarray, yhat: np.ndarray, sigma: np.ndarray, n_params: int) -> Dict[str, float]:
    resid = y - yhat
    rss = float(np.sum(resid**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    tss = float(np.sum((y - np.mean(y))**2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan

    chi2 = float(np.sum((resid / sigma)**2))
    dof = int(len(y) - n_params)
    red_chi2 = float(chi2 / dof) if dof > 0 else np.nan

    # For known pointwise Gaussian sigma, constants common to the compared
    # models cancel.  These are therefore useful *relative* AIC/BIC values.
    aic = float(chi2 + 2 * n_params)
    bic = float(chi2 + n_params * np.log(len(y)))
    return {
        "rss": rss,
        "rmse": rmse,
        "r2": r2,
        "chi2": chi2,
        "red_chi2": red_chi2,
        "aic": aic,
        "bic": bic,
        "dof": dof,
    }


def fit_scaling_models(
    t: np.ndarray,
    runs: np.ndarray,
    window: Tuple[float, float],
) -> List[Dict[str, object]]:
    """Fit log and offset-power laws to the ensemble mean in one time window.

    Models
    ------
    log:   y(t) = a + b log(t)
    power: y(t) = a + c t**alpha

    Fits are weighted with the ensemble SEM at each time.  The returned fit
    diagnostics are intended mainly for comparison between these two empirical
    scaling forms.  Because adjacent time samples are correlated, reduced chi^2,
    AIC and BIC should not be interpreted as formal independent-sample tests.
    """
    if curve_fit is None:
        raise RuntimeError("Scaling fits require scipy.optimize.curve_fit")

    stats = ensemble_stats(runs, "sem")
    mean = stats["mean"]
    sem = stats["sem"]
    lo, hi = window
    mask = (
        np.isfinite(t) & (t >= lo) & (t <= hi) & (t > 0)
        & np.isfinite(mean)
    )
    tt = np.asarray(t[mask], dtype=float)
    yy = np.asarray(mean[mask], dtype=float)
    ss = _safe_sigma(np.asarray(sem[mask], dtype=float), yy)

    if len(tt) < 6:
        return []

    results: List[Dict[str, object]] = []

    # Logarithmic model is linear in log(t); curve_fit gives covariance and
    # keeps the treatment consistent with the power-law model.
    p0_log = [float(np.median(yy)), float((yy[-1] - yy[0]) / max(np.log(tt[-1] / tt[0]), 1e-12))]
    try:
        popt, pcov = curve_fit(
            _log_model, tt, yy, p0=p0_log, sigma=ss,
            absolute_sigma=True, maxfev=50000,
        )
        perr = np.sqrt(np.diag(pcov))
        yhat = _log_model(tt, *popt)
        row = {
            "model": "log",
            "n_points": len(tt),
            "tmin_actual": float(tt.min()),
            "tmax_actual": float(tt.max()),
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "param_errors": {"a": float(perr[0]), "b": float(perr[1])},
            "success": True,
        }
        row.update(_goodness(yy, yhat, ss, 2))
        results.append(row)
    except Exception as exc:
        results.append({"model": "log", "success": False, "error": str(exc), "n_points": len(tt)})

    # Offset power law.  We constrain alpha to a broad positive range.  This
    # represents algebraic growth; a vanishing alpha is nearly degenerate with
    # logarithmic growth, so alpha and its uncertainty are important diagnostics.
    span = max(float(tt[-1] / tt[0]), 1.0)
    alpha0 = 0.25
    denom = max(tt[-1]**alpha0 - tt[0]**alpha0, 1e-12)
    c0 = float((yy[-1] - yy[0]) / denom)
    a0 = float(yy[0] - c0 * tt[0]**alpha0)
    scale = max(float(np.nanmax(np.abs(yy))), 1.0)
    try:
        popt, pcov = curve_fit(
            _power_model, tt, yy,
            p0=[a0, c0, alpha0],
            sigma=ss,
            absolute_sigma=True,
            bounds=([-1000 * scale, -1000 * scale, 1.0e-4],
                    [ 1000 * scale,  1000 * scale, 3.0]),
            maxfev=100000,
        )
        perr = np.sqrt(np.diag(pcov))
        yhat = _power_model(tt, *popt)
        row = {
            "model": "power",
            "n_points": len(tt),
            "tmin_actual": float(tt.min()),
            "tmax_actual": float(tt.max()),
            "params": {"a": float(popt[0]), "c": float(popt[1]), "alpha": float(popt[2])},
            "param_errors": {"a": float(perr[0]), "c": float(perr[1]), "alpha": float(perr[2])},
            "success": True,
        }
        row.update(_goodness(yy, yhat, ss, 3))
        results.append(row)
    except Exception as exc:
        results.append({"model": "power", "success": False, "error": str(exc), "n_points": len(tt)})

    return results


def run_scaling_analysis(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    observables: List[str],
    window_specs: List[str],
) -> List[Dict[str, object]]:
    """Fit all requested observables/conditions/windows and save diagnostics."""
    windows = [(spec, _parse_window(spec)) for spec in window_specs]
    rows: List[Dict[str, object]] = []

    report_lines = [
        "Scaling fits (ensemble mean weighted by SEM)",
        "Models: log y=a+b ln(t); power y=a+c t^alpha",
        "NOTE: time points are correlated, so chi^2/AIC/BIC are comparative diagnostics, not formal independent-sample tests.",
        "",
    ]
    print("\n" + "\n".join(report_lines))

    for condition, (t, metrics) in datasets.items():
        for obs in observables:
            if obs not in metrics:
                msg = f"  [skip] {condition}/{obs}: observable not present"
                print(msg)
                report_lines.append(msg)
                continue
            for spec, window in windows:
                fits = fit_scaling_models(t, metrics[obs], window)
                if not fits:
                    msg = f"  [skip] {condition}/{obs} window {spec}: too few points"
                    print(msg)
                    report_lines.append(msg)
                    continue

                successful = [r for r in fits if r.get("success")]
                if successful:
                    best_aic = min(float(r["aic"]) for r in successful)
                    best_bic = min(float(r["bic"]) for r in successful)
                else:
                    best_aic = best_bic = np.nan

                heading = f"{condition:18s}  {obs:30s}  t={spec}"
                print(heading)
                report_lines.append(heading)
                for fit in fits:
                    row = {
                        "condition": condition,
                        "observable": obs,
                        "window": spec,
                        **fit,
                    }
                    if fit.get("success"):
                        row["delta_aic"] = float(fit["aic"]) - best_aic
                        row["delta_bic"] = float(fit["bic"]) - best_bic
                        pars = fit["params"]
                        errs = fit["param_errors"]
                        if fit["model"] == "log":
                            ptxt = f"a={pars['a']:.6g}±{errs['a']:.2g}, b={pars['b']:.6g}±{errs['b']:.2g}"
                        else:
                            ptxt = (
                                f"a={pars['a']:.6g}±{errs['a']:.2g}, "
                                f"c={pars['c']:.6g}±{errs['c']:.2g}, "
                                f"alpha={pars['alpha']:.6g}±{errs['alpha']:.2g}"
                            )
                        msg = (
                            f"    {fit['model']:5s}: {ptxt}; "
                            f"R²={fit['r2']:.5f}, RMSE={fit['rmse']:.4g}, "
                            f"chi²_red={fit['red_chi2']:.4g}, "
                            f"AIC={fit['aic']:.3f} (Δ={row['delta_aic']:.3f}), "
                            f"BIC={fit['bic']:.3f} (Δ={row['delta_bic']:.3f})"
                        )
                        print(msg)
                        report_lines.append(msg)
                    else:
                        row["delta_aic"] = np.nan
                        row["delta_bic"] = np.nan
                        msg = f"    {fit['model']:5s}: FAILED: {fit.get('error', 'unknown error')}"
                        print(msg)
                        report_lines.append(msg)
                    rows.append(row)
                print()
                report_lines.append("")

    # Machine-readable diagnostics.
    csv_path = outdir / "scaling_fit_diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "condition", "observable", "window", "model", "success", "n_points",
            "tmin_actual", "tmax_actual",
            "a", "a_err", "b", "b_err", "c", "c_err", "alpha", "alpha_err",
            "r2", "rmse", "rss", "chi2", "red_chi2", "dof", "aic", "delta_aic",
            "bic", "delta_bic", "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pars = row.get("params", {}) or {}
            errs = row.get("param_errors", {}) or {}
            flat = {
                "condition": row.get("condition"),
                "observable": row.get("observable"),
                "window": row.get("window"),
                "model": row.get("model"),
                "success": row.get("success"),
                "n_points": row.get("n_points"),
                "tmin_actual": row.get("tmin_actual"),
                "tmax_actual": row.get("tmax_actual"),
                "a": pars.get("a"), "a_err": errs.get("a"),
                "b": pars.get("b"), "b_err": errs.get("b"),
                "c": pars.get("c"), "c_err": errs.get("c"),
                "alpha": pars.get("alpha"), "alpha_err": errs.get("alpha"),
                "r2": row.get("r2"), "rmse": row.get("rmse"), "rss": row.get("rss"),
                "chi2": row.get("chi2"), "red_chi2": row.get("red_chi2"),
                "dof": row.get("dof"), "aic": row.get("aic"), "delta_aic": row.get("delta_aic"),
                "bic": row.get("bic"), "delta_bic": row.get("delta_bic"),
                "error": row.get("error"),
            }
            writer.writerow(flat)

    (outdir / "scaling_fit_report.txt").write_text("\n".join(report_lines) + "\n")
    return rows


def plot_fit_diagnostics(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    fit_rows: List[Dict[str, object]],
    observables: List[str],
    plot_window: str,
    error_mode: str,
) -> None:
    """Show mean data and both candidate scaling fits for one selected window."""
    import matplotlib.pyplot as plt

    if not observables:
        return
    chosen = _parse_window(plot_window)
    nobs = len(observables)
    fig, axes = plt.subplots(1, nobs, figsize=(6.0 * nobs, 4.6), constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    ylabels = {
        "interface_ratio_sigma1": r"geometric interface / initial ($\sigma=1$)",
        "n_domains": "number of A/B domains",
        "voter_interface_ratio": "legacy interface / initial",
    }

    for ax, obs in zip(axes, observables):
        for condition, (t, metrics) in datasets.items():
            if obs not in metrics:
                continue
            label = CONDITION_LABELS.get(condition, condition)
            line = _plot_band(ax, t, metrics[obs], label, error_mode)

            # Overlay fits only when the row exactly matches the requested window.
            matching = [
                r for r in fit_rows
                if r.get("condition") == condition
                and r.get("observable") == obs
                and r.get("window") == plot_window
                and r.get("success")
            ]
            tt = np.geomspace(chosen[0], chosen[1], 250)
            for r in matching:
                pars = r["params"]
                if r["model"] == "log":
                    yy = _log_model(tt, pars["a"], pars["b"])
                    ls = "--"
                else:
                    yy = _power_model(tt, pars["a"], pars["c"], pars["alpha"])
                    ls = ":"
                ax.plot(tt, yy, ls=ls, lw=1.25, color=line.get_color(), alpha=0.9)

        _style_time_axis(ax)
        ax.set_xlim(chosen)
        ax.set_ylabel(ylabels.get(obs, obs))
        ax.set_title(obs.replace("_", " "))

    # Legend: condition colors plus model line styles.
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], color="0.25", ls="--", lw=1.25, label="log fit"),
        Line2D([0], [0], color="0.25", ls=":", lw=1.25, label="power fit"),
    ]
    labels += ["log fit", "power fit"]
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(f"Scaling-fit diagnostics, t={plot_window}")
    fig.savefig(outdir / "07_scaling_fit_diagnostics.png", dpi=240)
    plt.close(fig)


def plot_paper_interface_fragmentation(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """Paper-ready two-panel summary: relative interface + fragmentation."""
    import matplotlib.pyplot as plt

    rc = {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=True)

        for condition, (t, metrics) in datasets.items():
            label = CONDITION_LABELS.get(condition, condition)
            if "interface_ratio_sigma1" in metrics:
                _plot_band(axes[0], t, metrics["interface_ratio_sigma1"], label, error_mode)
            if "n_domains" in metrics:
                _plot_band(axes[1], t, metrics["n_domains"], label, error_mode)

        for ax in axes:
            _style_time_axis(ax)
            ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
            ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].axhline(1.0, ls="--", lw=0.9, color="0.5", alpha=0.75, zorder=0)
        axes[0].set_ylabel(r"$\langle\mathcal{L}_{\sigma=1}(t)/\mathcal{L}_{\sigma=1}(0)\rangle$")
        axes[1].set_ylabel(r"$\langle N_{\rm domains}(t)\rangle$")

        axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top", ha="left", fontweight="bold")
        axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top", ha="left", fontweight="bold")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=len(labels), frameon=False)

        png = outdir / "08_paper_interface_fragmentation.png"
        pdf = outdir / "08_paper_interface_fragmentation.pdf"
        fig.savefig(png, dpi=600, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Effective logarithmic growth rate: b_eff = t dI/dt = dI/d(ln t)
# -----------------------------------------------------------------------------
def local_log_slopes(
    time: np.ndarray,
    runs: np.ndarray,
    *,
    halfwidth_decades: float = 0.2,
    n_centers: int = 100,
    min_points: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate dI/dln(t) in centered, sliding logarithmic time windows.

    The window at t_c is [t_c / 10**halfwidth, t_c * 10**halfwidth].
    At every center, a weighted local linear regression of I against ln(t) is
    performed *for each realization*.  Trapezoidal weights in ln(t) prevent
    the linearly spaced simulation frames from over-weighting late times.
    These are descriptive local slopes, not independent time samples.

    Returns
    -------
    centers : (n_centers,) times, restricted to complete centered windows
    slopes : (n_runs, n_centers) run-level slopes (NaN if under-resolved)
    counts : (n_centers,) number of raw simulation frames in the local window
    """
    time = np.asarray(time, dtype=float)
    runs = np.asarray(runs, dtype=float)
    if runs.ndim != 2 or runs.shape[1] != time.size:
        raise ValueError("runs must have shape (n_runs, len(time))")
    if not np.isfinite(halfwidth_decades) or halfwidth_decades <= 0:
        raise ValueError("halfwidth_decades must be positive and finite")
    if n_centers < 2 or min_points < 3:
        raise ValueError("n_centers must be >= 2 and min_points >= 3")

    valid = np.isfinite(time) & (time > 0)
    tt = time[valid]
    values = runs[:, valid]
    if tt.size < min_points or np.any(np.diff(tt) <= 0):
        raise ValueError("positive time points must be strictly increasing and sufficiently many")

    factor = 10.0 ** halfwidth_decades
    t_lo = tt[0] * factor
    t_hi = tt[-1] / factor
    if t_lo >= t_hi:
        raise ValueError("time range too short for centered windows of requested width")

    centers = np.geomspace(t_lo, t_hi, n_centers)
    x = np.log(tt)
    log_halfwidth = halfwidth_decades * np.log(10.0)
    slopes = np.full((runs.shape[0], n_centers), np.nan, dtype=float)
    counts = np.zeros(n_centers, dtype=int)

    for i, center in enumerate(centers):
        xc = np.log(center)
        window = np.abs(x - xc) <= log_halfwidth * (1.0 + 1e-12)
        xx = x[window]
        counts[i] = xx.size
        if xx.size < min_points:
            continue

        # Trapezoidal quadrature in x=ln(t): equal influence per logarithmic
        # interval, even though the input frames are approximately uniform in t.
        dx = np.diff(xx)
        weights = np.empty(xx.size, dtype=float)
        weights[0] = dx[0] / 2
        weights[-1] = dx[-1] / 2
        weights[1:-1] = (dx[:-1] + dx[1:]) / 2
        weights *= np.maximum(0.0, 1.0 - np.abs(xx - xc) / log_halfwidth)

        for irun, yy in enumerate(values[:, window]):
            good = np.isfinite(yy) & (weights > 0)
            if np.count_nonzero(good) < min_points:
                continue
            w = weights[good]
            xx_good = xx[good]
            yy_good = yy[good]
            xbar = np.average(xx_good, weights=w)
            ybar = np.average(yy_good, weights=w)
            centered = xx_good - xbar
            denom = np.dot(w, centered**2)
            if denom <= 0:
                continue
            slopes[irun, i] = np.dot(w, centered * (yy_good - ybar)) / denom

    return centers, slopes, counts


def compute_local_slope_datasets(
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    observables: List[str],
    *,
    halfwidth_decades: float,
    n_centers: int,
    min_points: int,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Return per-condition, per-observable local slopes for plots and CSV."""
    derived: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for condition, (t, metrics) in datasets.items():
        derived[condition] = {}
        for obs in observables:
            if obs not in metrics:
                continue
            derived[condition][obs] = local_log_slopes(
                t, metrics[obs],
                halfwidth_decades=halfwidth_decades,
                n_centers=n_centers,
                min_points=min_points,
            )
    return derived


def save_local_slope_csv(
    outdir: Path,
    derived: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    *,
    error_mode: str,
    halfwidth_decades: float,
    min_points: int,
) -> None:
    """Export all slopes, including error bands computed across independent runs."""
    path = outdir / "effective_log_slopes.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "observable", "time_center", "window_tmin", "window_tmax",
            "n_frames", "n_runs", "b_eff_mean", "b_eff_std", "b_eff_sem",
            "error_halfwidth", "error_mode", "halfwidth_decades", "min_points",
        ])
        factor = 10.0 ** halfwidth_decades
        for condition, metrics in derived.items():
            for obs, (centers, slopes, counts) in metrics.items():
                valid_centers = (counts >= min_points) & (
                    np.sum(np.isfinite(slopes), axis=0) > 0
                )
                if not np.any(valid_centers):
                    continue
                s = ensemble_stats(slopes[:, valid_centers], error_mode)
                for i, center in enumerate(centers[valid_centers]):
                    writer.writerow([
                        condition, obs, float(center), float(center / factor),
                        float(center * factor), int(counts[valid_centers][i]),
                        int(s["n"][i]), float(s["mean"][i]),
                        float(s["std"][i]), float(s["sem"][i]),
                        float(s["err"][i]), error_mode, halfwidth_decades, min_points,
                    ])


def plot_local_log_slopes(
    outdir: Path,
    derived: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    fit_rows: List[Dict[str, object]],
    *,
    observables: List[str],
    reference_window: str,
    error_mode: str,
    halfwidth_decades: float,
    min_points: int,
) -> None:
    """Plot run-averaged b_eff(t), alongside the late-time log-fit slopes."""
    import matplotlib.pyplot as plt

    labels = {
        "interface_ratio_sigma1": r"relative geometric interface ($\sigma=1$)",
        "voter_interface_ratio": "relative correlation-based interface",
        "n_domains": "number of connected domains",
    }
    available = [obs for obs in observables if any(obs in d for d in derived.values())]
    if not available:
        return
    fig, axes = plt.subplots(
        1, len(available), figsize=(5.1 * len(available), 4.3),
        constrained_layout=True, squeeze=False,
    )
    reference_lo, reference_hi = _parse_window(reference_window)

    for ax, obs in zip(axes.ravel(), available):
        for condition, metric_dict in derived.items():
            if obs not in metric_dict:
                continue
            centers, slopes, counts = metric_dict[obs]
            usable = counts >= min_points
            usable &= np.sum(np.isfinite(slopes), axis=0) > 0
            if not np.any(usable):
                continue
            stats = ensemble_stats(slopes[:, usable], error_mode)
            xx = centers[usable]
            yy = stats["mean"]
            err = stats["err"]
            line, = ax.plot(xx, yy, lw=1.7, label=CONDITION_LABELS.get(condition, condition))
            ax.fill_between(
                xx, yy - err, yy + err, color=line.get_color(), alpha=0.18,
                linewidth=0,
            )

            # Compare a local slope directly against b of a late-time global
            # fit I=a+b ln(t), if a matching fit was requested and computed.
            for fit in fit_rows:
                if (
                    fit.get("condition") == condition
                    and fit.get("observable") == obs
                    and fit.get("window") == reference_window
                    and fit.get("model") == "log"
                    and fit.get("success")
                ):
                    b = float(fit["params"]["b"])
                    refmask = (xx >= reference_lo) & (xx <= reference_hi)
                    if np.any(refmask):
                        ax.plot(
                            xx[refmask], np.full(np.count_nonzero(refmask), b),
                            color=line.get_color(), ls="--", lw=1.3, alpha=0.85,
                            label=f"{CONDITION_LABELS.get(condition, condition)}: log-fit slope",
                        )
                    break

        _style_time_axis(ax)
        ax.set_title(labels.get(obs, obs))
        ax.set_ylabel(r"$b_{\mathrm{eff}}(t)=dI/d\ln t$")
        ax.axhline(0.0, color="0.45", lw=0.8, alpha=0.7)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        rf"Effective logarithmic growth rate (local fits, half-width {halfwidth_decades:g} decades)"
    )
    fig.savefig(outdir / "09_effective_log_slope.png", dpi=300)
    fig.savefig(outdir / "09_effective_log_slope.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
def write_long_csv(
    output_path: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """Write one tidy row per condition / time / observable."""
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition",
            "time",
            "observable",
            "n_runs",
            "mean",
            "std",
            "sem",
            "error_halfwidth",
            "error_mode",
        ])

        for condition, (time, metrics) in datasets.items():
            for name in sorted(metrics):
                s = ensemble_stats(metrics[name], error_mode)
                for i, t in enumerate(time):
                    writer.writerow([
                        condition,
                        float(t),
                        name,
                        int(s["n"][i]),
                        float(s["mean"][i]),
                        float(s["std"][i]),
                        float(s["sem"][i]),
                        float(s["err"][i]),
                        error_mode,
                    ])


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _positive_time_mask(t: np.ndarray) -> np.ndarray:
    return np.isfinite(t) & (t > 0)


def _style_time_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("time")
    ax.grid(True, which="both", alpha=0.2)


def _plot_band(ax, t, x, label, error_mode, **plot_kwargs):
    s = ensemble_stats(x, error_mode)
    mask = _positive_time_mask(t) & np.isfinite(s["mean"])
    line, = ax.plot(t[mask], s["mean"][mask], label=label, **plot_kwargs)
    ylo = s["mean"] - s["err"]
    yhi = s["mean"] + s["err"]
    ax.fill_between(
        t[mask], ylo[mask], yhi[mask],
        alpha=0.18,
        color=line.get_color(),
        linewidth=0,
    )
    return line


def plot_geometric_by_scale(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """One panel per smoothing scale; conditions overlaid."""
    import matplotlib.pyplot as plt

    # Union of scales across conditions.
    all_names = set()
    for _, metrics in datasets.values():
        all_names.update(geometric_interface_names(metrics))
    names = sorted(
        all_names,
        key=lambda x: float(x.removeprefix("interface_density_sigma")),
    )
    if not names:
        return

    ncols = 2
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        sigma = name.removeprefix("interface_density_sigma")
        for condition, (t, metrics) in datasets.items():
            if name not in metrics:
                continue
            _plot_band(
                ax, t, metrics[name],
                CONDITION_LABELS.get(condition, condition),
                error_mode,
            )
        _style_time_axis(ax)
        ax.set_ylabel(r"interface length density $\mathcal{L}/A$")
        ax.set_title(rf"geometric interface, $\sigma={sigma}$ cells")

    for ax in axes[len(names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Geometric interface density: ensemble mean with uncertainty")
    fig.savefig(outdir / "01_geometric_interface_density_logtime.png", dpi=220)
    plt.close(fig)


def plot_multiscale_per_condition(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """Within each condition, compare all geometric smoothing scales."""
    import matplotlib.pyplot as plt

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.6), constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    plotted_any = False
    for ax, (condition, (t, metrics)) in zip(axes, datasets.items()):
        names = geometric_interface_names(metrics)
        for name in names:
            sigma = name.removeprefix("interface_density_sigma")
            _plot_band(ax, t, metrics[name], rf"$\sigma={sigma}$", error_mode)
            plotted_any = True
        _style_time_axis(ax)
        ax.set_ylabel(r"interface length density $\mathcal{L}/A$")
        ax.set_title(CONDITION_LABELS.get(condition, condition))
        if names:
            ax.legend()

    if plotted_any:
        fig.suptitle("Scale dependence of geometric interface density")
        fig.savefig(outdir / "02_multiscale_interface_density_logtime.png", dpi=220)
    plt.close(fig)


def plot_legacy_interface(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    any_data = False
    for condition, (t, metrics) in datasets.items():
        if "voter_interface_density" not in metrics:
            continue
        _plot_band(
            ax, t, metrics["voter_interface_density"],
            CONDITION_LABELS.get(condition, condition),
            error_mode,
        )
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    _style_time_axis(ax)
    ax.set_ylabel(r"legacy interface density $1-\langle m_i m_j\rangle$")
    ax.legend()
    ax.set_title("Nearest-neighbour correlation interface measure")
    fig.savefig(outdir / "03_legacy_voter_interface_density_logtime.png", dpi=220)
    plt.close(fig)


def plot_normalized_breakup(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """Compare normalized geometric and legacy interface growth."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for condition, (t, metrics) in datasets.items():
        label = CONDITION_LABELS.get(condition, condition)
        if "interface_ratio_sigma1" in metrics:
            _plot_band(axes[0], t, metrics["interface_ratio_sigma1"], label, error_mode)
        if "voter_interface_ratio" in metrics:
            _plot_band(axes[1], t, metrics["voter_interface_ratio"], label, error_mode)

    for ax in axes:
        _style_time_axis(ax)
        ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.legend()

    axes[0].set_ylabel(r"geometric interface / initial ($\sigma=1$)")
    axes[0].set_title("Geometric breakup")
    axes[1].set_ylabel("legacy interface / initial")
    axes[1].set_title("Correlation-based breakup")

    fig.suptitle("Relative growth of the interface")
    fig.savefig(outdir / "04_normalized_interface_breakup_logtime.png", dpi=220)
    plt.close(fig)


def plot_all_geometric_ratios(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """One panel per condition with geometric perimeter normalized at each scale."""
    import matplotlib.pyplot as plt

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.6), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    any_data = False

    for ax, (condition, (t, metrics)) in zip(axes, datasets.items()):
        ratio_names = [
            k for k in metrics
            if k.startswith("interface_ratio_sigma")
        ]
        ratio_names = sorted(
            ratio_names,
            key=lambda x: float(x.removeprefix("interface_ratio_sigma")),
        )
        for name in ratio_names:
            sigma = name.removeprefix("interface_ratio_sigma")
            _plot_band(ax, t, metrics[name], rf"$\sigma={sigma}$", error_mode)
            any_data = True
        _style_time_axis(ax)
        ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_ylabel("geometric interface / initial")
        ax.set_title(CONDITION_LABELS.get(condition, condition))
        if ratio_names:
            ax.legend()

    if any_data:
        fig.suptitle("Normalized geometric interface at all coarse-graining scales")
        fig.savefig(outdir / "05_geometric_interface_ratios_logtime.png", dpi=220)
    plt.close(fig)


def plot_roughness_and_fragmentation(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """Secondary interface/fragmentation measures used in the ensemble driver."""
    import matplotlib.pyplot as plt

    panels = [
        ("total_variation", "total variation", "smoothed composition TV"),
        ("effective_boundary_dimension", r"effective $D_{\rm eff}$", "multiscale boundary dimension"),
        ("roughness_ratio", r"$P(\sigma=0)/P(\sigma=4)$", "boundary roughness ratio"),
        ("n_domains", "number of A/B domains", "domain fragmentation"),
        ("largest_domain_fraction", "largest-domain area fraction", "largest connected domain"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (name, ylabel, title) in zip(axes, panels):
        has_data = False
        for condition, (t, metrics) in datasets.items():
            if name not in metrics:
                continue
            _plot_band(
                ax, t, metrics[name],
                CONDITION_LABELS.get(condition, condition),
                error_mode,
            )
            has_data = True
        _style_time_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if has_data:
            ax.legend()

    axes[-1].axis("off")
    fig.suptitle("Interface roughness and fragmentation diagnostics")
    fig.savefig(outdir / "06_roughness_fragmentation_logtime.png", dpi=220)
    plt.close(fig)


def plot_overview(
    outdir: Path,
    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    error_mode: str,
) -> None:
    """Compact 2x3 figure suitable for quickly judging the experiment."""
    import matplotlib.pyplot as plt

    panels = [
        ("interface_density_sigma1", r"$\mathcal{L}/A$", r"geometric interface ($\sigma=1$)"),
        ("voter_interface_density", r"$1-\langle m_i m_j\rangle$", "legacy interface density"),
        ("interface_ratio_sigma1", "interface / initial", "relative geometric breakup"),
        ("effective_boundary_dimension", r"$D_{\rm eff}$", "boundary dimension"),
        ("roughness_ratio", r"$P_0/P_4$", "roughness ratio"),
        ("n_domains", "number of domains", "fragmentation"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (name, ylabel, title) in zip(axes, panels):
        for condition, (t, metrics) in datasets.items():
            if name not in metrics:
                continue
            _plot_band(
                ax, t, metrics[name],
                CONDITION_LABELS.get(condition, condition),
                error_mode,
            )
        _style_time_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Schelling interface breakup: ensemble averages")
    fig.savefig(outdir / "00_interface_overview_logtime.png", dpi=240)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("interface_ensemble"),
        help="root output directory created by interface_ensemble.py",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["sv_stoc", "schelling_stoc"],
        help="subdirectories/conditions to compare",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="analysis output directory (default: INPUT/analysis)",
    )
    p.add_argument(
        "--error",
        choices=("ci95", "sem", "std"),
        default="ci95",
        help="uncertainty band shown around ensemble mean",
    )
    p.add_argument(
        "--fit-observables",
        nargs="+",
        default=["interface_ratio_sigma1", "voter_interface_ratio", "n_domains"],
        help="observables for log-vs-power scaling analysis",
    )
    p.add_argument(
        "--fit-windows",
        nargs="+",
        default=["1:20", "20:200", "200:1000", "1:1000"],
        help="fit windows as TMIN:TMAX (positive times only)",
    )
    p.add_argument(
        "--fit-plot-window",
        default="1:1000",
        help="one of --fit-windows to overlay in the scaling diagnostic figure",
    )
    p.add_argument(
        "--no-fits",
        action="store_true",
        help="skip scaling fits and fit diagnostic plot",
    )
    p.add_argument(
        "--slope-observables",
        nargs="+",
        default=["interface_ratio_sigma1", "voter_interface_ratio", "n_domains"],
        help="observables for local logarithmic slope plot and CSV",
    )
    p.add_argument(
        "--slope-halfwidth-decades",
        type=float,
        default=0.2,
        help="half-width of centered local regression window in log10(time) (default: 0.2)",
    )
    p.add_argument(
        "--slope-centers",
        type=int,
        default=100,
        help="number of logarithmically spaced centers for local slope regression",
    )
    p.add_argument(
        "--slope-min-points",
        type=int,
        default=8,
        help="minimum number of time samples per local regression (default: 8)",
    )
    p.add_argument(
        "--slope-reference-window",
        default="200:1000",
        help="window whose global log-fit slope is drawn as dashed horizontal reference",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input
    outdir = args.output if args.output is not None else root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}

    print(f"Reading ensemble from: {root}")
    for condition in args.conditions:
        cdir = root / condition
        time, metrics, files = load_condition(cdir)
        datasets[condition] = (time, metrics)
        print(
            f"  {condition:18s}: {len(files):3d} runs, "
            f"{len(time):4d} frames, t=[{time[0]:g}, {time[-1]:g}]"
        )

    csv_path = outdir / "ensemble_interface_timeseries.csv"
    write_long_csv(csv_path, datasets, args.error)

    # Plotting imports are kept out of worker/simulation code and happen only here.
    plot_overview(outdir, datasets, args.error)
    plot_geometric_by_scale(outdir, datasets, args.error)
    plot_multiscale_per_condition(outdir, datasets, args.error)
    plot_legacy_interface(outdir, datasets, args.error)
    plot_normalized_breakup(outdir, datasets, args.error)
    plot_all_geometric_ratios(outdir, datasets, args.error)
    plot_roughness_and_fragmentation(outdir, datasets, args.error)
    plot_paper_interface_fragmentation(outdir, datasets, args.error)

    fit_rows = []
    if not args.no_fits:
        if args.fit_plot_window not in args.fit_windows:
            print(
                f"WARNING: --fit-plot-window {args.fit_plot_window!r} is not present in "
                "--fit-windows; adding it for the diagnostic plot."
            )
            args.fit_windows = list(args.fit_windows) + [args.fit_plot_window]
        fit_rows = run_scaling_analysis(
            outdir, datasets, args.fit_observables, args.fit_windows
        )
        plot_fit_diagnostics(
            outdir, datasets, fit_rows, args.fit_observables,
            args.fit_plot_window, args.error,
        )

    # Local logarithmic derivatives are calculated separately for each run,
    # so the uncertainty bands reflect between-realization variation rather
    # than treating neighboring times as independent observations.
    slope_datasets = compute_local_slope_datasets(
        datasets, args.slope_observables,
        halfwidth_decades=args.slope_halfwidth_decades,
        n_centers=args.slope_centers,
        min_points=args.slope_min_points,
    )
    save_local_slope_csv(
        outdir, slope_datasets, error_mode=args.error,
        halfwidth_decades=args.slope_halfwidth_decades,
        min_points=args.slope_min_points,
    )
    plot_local_log_slopes(
        outdir, slope_datasets, fit_rows, observables=args.slope_observables,
        reference_window=args.slope_reference_window, error_mode=args.error,
        halfwidth_decades=args.slope_halfwidth_decades,
        min_points=args.slope_min_points,
    )

    print(f"\nWrote analysis to: {outdir}")
    print(f"  CSV: {csv_path.name}")
    print(f"  Error bands: {args.error}")
    if not args.no_fits:
        print("  Scaling diagnostics: scaling_fit_diagnostics.csv + scaling_fit_report.txt")
        print("  Paper figure: 08_paper_interface_fragmentation.pdf (+ 600 dpi PNG)")
    print("  Local slopes: effective_log_slopes.csv + 09_effective_log_slope.png/pdf")
    print(
        "  b_eff is fitted per run in centered log-time windows; the error bands "
        "are across runs (not independent-time fit errors)."
    )
    print("  All figure x-axes use logarithmic time; t=0 is retained in the CSV but not plotted.")


if __name__ == "__main__":
    main()
