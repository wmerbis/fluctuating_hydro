#!/usr/bin/env python3
"""
Summarize projection diagnostics for Schelling-Voter sweeps.

Supports both:
- projection_mode="clip"
- projection_mode="redistribute"

For redistribute runs, the headline diagnostics are:
- cumulative relative mass redistributed, split into
    low/high/simplex x local/global
- fraction of redistributed mass handled globally
- redistribution failure counts
- local-vs-global redistribution event counts inferred from history increments
- global failure counts and failure fractions

Expected files
--------------
data/{regime}/<base>_projection_diag.json
data/{regime}/<base>_projection_history.csv

Default base pattern:
VS_run_{n_run}_det_Dv{D_v:.3f}

Examples
--------
Deterministic:
    python summarize_projection_diagnostics.py

Stochastic:
    python summarize_projection_diagnostics.py \
        --input-pattern 'VS_run_{n_run}_sto_Dv{D_v:.3f}' \
        --output-dir projection_summary_stochastic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


REGIMES = ["segregating", "integrating", "migrating", "well-mixed"]
DV_VALS = np.linspace(0.01, 0.2, 20)
N_RUNS = 10
EPS = 1e-14

REDIST_MASS_FIELDS = [
    "mass_redistributed_low_local",
    "mass_redistributed_low_global",
    "mass_redistributed_high_local",
    "mass_redistributed_high_global",
    "mass_redistributed_simplex_local",
    "mass_redistributed_simplex_global",
]

REDIST_FAILURE_FIELDS = [
    "n_redistribute_low_failures",
    "n_redistribute_high_failures",
    "n_redistribute_simplex_failures",
]

REDIST_LEFTOVER_FIELDS = [
    "redistribute_low_leftover",
    "redistribute_high_leftover",
    "redistribute_simplex_leftover",
]

KEY_SUMMARY_FIELDS = [
    "n_calls",
    "n_expensive_projection_calls",
    "n_roundoff_cleanup_calls",
    "net_mass_change_projection",
    "mass_before_projection",
    "mass_after_projection",
    *REDIST_MASS_FIELDS,
    *REDIST_LEFTOVER_FIELDS,
    *REDIST_FAILURE_FIELDS,
    "n_low_entries",
    "n_high_entries",
    "n_simplex_cells",
    "mass_added_low",
    "mass_removed_high",
    "mass_removed_simplex",
    "max_low_violation",
    "max_high_violation",
    "max_simplex_violation",
    "mass_roundoff_cleanup",
]

CUMULATIVE_HISTORY_FIELDS = REDIST_MASS_FIELDS


def as_float(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def render_pattern(pattern, regime, n_run, dv):
    return pattern.format(
        regime=regime,
        n_run=n_run,
        run=n_run,
        D_v=dv,
        Dv=dv,
        dv=dv,
    )


def read_json_diag(path):
    with path.open("r") as f:
        obj = json.load(f)
    return obj.get("metadata", {}), obj.get("projection_summary", {})


def read_history(path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            d = {"_row_index": i}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                fv = as_float(v)
                d[k] = fv if fv is not None else v
            rows.append(d)
    return rows


def infer_reference_mass(metadata, summary, history):
    """
    Prefer an explicitly stored initial/reference mass. Otherwise use the
    first history record's mass_before. This is the best available run-specific
    normalization for cumulative redistributed mass.
    """
    for source in (metadata, summary):
        for key in ("initial_total_mass", "initial_mass", "mass_initial", "M_initial"):
            v = as_float(source.get(key))
            if v is not None and abs(v) > EPS:
                return v

    if history:
        v = as_float(history[0].get("mass_before"))
        if v is not None and abs(v) > EPS:
            return v

    return None


def diff_cumulative_history(rows):
    """
    Add per-record increments d_<field> for cumulative redistribution masses.

    The first increment is defined relative to zero, because the cumulative
    diagnostic starts from zero at initialization.
    """
    if not rows:
        return rows

    prev = {f: 0.0 for f in CUMULATIVE_HISTORY_FIELDS}
    out = []

    for row in rows:
        r = dict(row)
        for field in CUMULATIVE_HISTORY_FIELDS:
            cur = as_float(row.get(field))
            if cur is None:
                continue
            delta = cur - prev[field]
            # Small negative deltas should only be roundoff; retain material ones.
            if delta < 0 and abs(delta) < 1e-12 * max(1.0, abs(cur), abs(prev[field])):
                delta = 0.0
            r[f"d_{field}"] = delta
            prev[field] = cur
        out.append(r)

    return out


def add_history_derived_fields(rows, reference_mass):
    """
    Add interval-relative redistributed masses and inferred local/global
    redistribution event indicators.

    Since history does not store explicit local/global call counters, an event
    is inferred when the corresponding cumulative mass increases during that
    record interval.
    """
    rows = diff_cumulative_history(rows)

    for r in rows:
        # Relative mass increments.
        for field in CUMULATIVE_HISTORY_FIELDS:
            dfield = f"d_{field}"
            val = as_float(r.get(dfield))
            if val is not None and reference_mass is not None:
                r[f"rel_{dfield}"] = val / reference_mass

        # Totals by locality and violation type.
        local_fields = [
            "d_mass_redistributed_low_local",
            "d_mass_redistributed_high_local",
            "d_mass_redistributed_simplex_local",
        ]
        global_fields = [
            "d_mass_redistributed_low_global",
            "d_mass_redistributed_high_global",
            "d_mass_redistributed_simplex_global",
        ]

        local_mass = sum(as_float(r.get(f)) or 0.0 for f in local_fields)
        global_mass = sum(as_float(r.get(f)) or 0.0 for f in global_fields)
        r["d_mass_redistributed_local_total"] = local_mass
        r["d_mass_redistributed_global_total"] = global_mass
        r["d_mass_redistributed_total"] = local_mass + global_mass

        if reference_mass is not None:
            r["rel_d_mass_redistributed_local_total"] = local_mass / reference_mass
            r["rel_d_mass_redistributed_global_total"] = global_mass / reference_mass
            r["rel_d_mass_redistributed_total"] = (local_mass + global_mass) / reference_mass

        # Inferred event counts from positive redistributed mass increments.
        for kind in ("low", "high", "simplex"):
            dl = as_float(r.get(f"d_mass_redistributed_{kind}_local")) or 0.0
            dg = as_float(r.get(f"d_mass_redistributed_{kind}_global")) or 0.0
            r[f"event_{kind}_local"] = float(dl > EPS)
            r[f"event_{kind}_global"] = float(dg > EPS)

        r["event_local_any"] = float(any(
            (as_float(r.get(f"d_mass_redistributed_{kind}_local")) or 0.0) > EPS
            for kind in ("low", "high", "simplex")
        ))
        r["event_global_any"] = float(any(
            (as_float(r.get(f"d_mass_redistributed_{kind}_global")) or 0.0) > EPS
            for kind in ("low", "high", "simplex")
        ))

    return rows


def add_summary_derived(summary, metadata, history):
    out = dict(summary)
    ref_mass = infer_reference_mass(metadata, summary, history)
    out["reference_mass"] = ref_mass if ref_mass is not None else np.nan

    # Relative cumulative redistributed masses.
    for field in REDIST_MASS_FIELDS:
        v = as_float(summary.get(field))
        if v is not None and ref_mass is not None:
            out[f"relative_{field}"] = v / ref_mass

    low_local = as_float(summary.get("mass_redistributed_low_local")) or 0.0
    low_global = as_float(summary.get("mass_redistributed_low_global")) or 0.0
    high_local = as_float(summary.get("mass_redistributed_high_local")) or 0.0
    high_global = as_float(summary.get("mass_redistributed_high_global")) or 0.0
    simplex_local = as_float(summary.get("mass_redistributed_simplex_local")) or 0.0
    simplex_global = as_float(summary.get("mass_redistributed_simplex_global")) or 0.0

    total_local = low_local + high_local + simplex_local
    total_global = low_global + high_global + simplex_global
    total_redist = total_local + total_global

    out["mass_redistributed_local_total"] = total_local
    out["mass_redistributed_global_total"] = total_global
    out["mass_redistributed_total"] = total_redist

    if ref_mass is not None:
        out["relative_mass_redistributed_local_total"] = total_local / ref_mass
        out["relative_mass_redistributed_global_total"] = total_global / ref_mass
        out["relative_mass_redistributed_total"] = total_redist / ref_mass

    out["global_redistributed_mass_fraction"] = (
        total_global / total_redist if total_redist > EPS else 0.0
    )

    # Failure counts.
    n_fail_low = as_float(summary.get("n_redistribute_low_failures")) or 0.0
    n_fail_high = as_float(summary.get("n_redistribute_high_failures")) or 0.0
    n_fail_simplex = as_float(summary.get("n_redistribute_simplex_failures")) or 0.0
    out["n_redistribute_failures_total"] = n_fail_low + n_fail_high + n_fail_simplex

    # Infer event counts from history mass increments.
    local_events = 0
    global_events = 0
    local_by_kind = defaultdict(int)
    global_by_kind = defaultdict(int)

    for r in history:
        for kind in ("low", "high", "simplex"):
            dl = as_float(r.get(f"d_mass_redistributed_{kind}_local")) or 0.0
            dg = as_float(r.get(f"d_mass_redistributed_{kind}_global")) or 0.0
            if dl > EPS:
                local_by_kind[kind] += 1
                local_events += 1
            if dg > EPS:
                global_by_kind[kind] += 1
                global_events += 1

    for kind in ("low", "high", "simplex"):
        out[f"n_redistribute_{kind}_local_events"] = local_by_kind[kind]
        out[f"n_redistribute_{kind}_global_events"] = global_by_kind[kind]

    out["n_redistribute_local_events_total"] = local_events
    out["n_redistribute_global_events_total"] = global_events

    # This is an event-based fallback fraction, not a true call fraction.
    # Exact call counts are not stored in the provided diagnostics.
    out["global_fallback_event_fraction"] = (
        global_events / (local_events + global_events)
        if (local_events + global_events) > 0 else 0.0
    )

    failures = out["n_redistribute_failures_total"]
    out["global_failure_per_global_event"] = (
        failures / global_events if global_events > 0 else (0.0 if failures == 0 else np.nan)
    )

    return out, ref_mass


def numeric_fields(rows):
    fields = set()
    for r in rows:
        for k, v in r.items():
            if k.startswith("_"):
                continue
            if as_float(v) is not None:
                fields.add(k)
    return sorted(fields)


def aggregate_history(run_histories):
    """
    Align histories by 'step' and calculate mean/std/count for numeric fields.
    """
    if not run_histories:
        return np.array([]), {}

    buckets = defaultdict(lambda: defaultdict(list))
    xs = set()

    for rows in run_histories:
        for j, r in enumerate(rows):
            x = as_float(r.get("step"))
            if x is None:
                x = float(j)
            xs.add(x)
            for field, value in r.items():
                if field in ("step", "_row_index"):
                    continue
                v = as_float(value)
                if v is not None:
                    buckets[field][x].append(v)

    xs = np.array(sorted(xs), dtype=float)
    stats = {}

    for field, byx in buckets.items():
        mean = np.full(len(xs), np.nan)
        std = np.full(len(xs), np.nan)
        count = np.zeros(len(xs), dtype=int)
        for i, x in enumerate(xs):
            vals = np.asarray(byx.get(x, []), dtype=float)
            if len(vals):
                mean[i] = vals.mean()
                std[i] = vals.std(ddof=0)
                count[i] = len(vals)
        stats[field] = (mean, std, count)

    return xs, stats


def write_long_history(path, aggregated):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "D_v", "step", "diagnostic", "mean", "std", "n_runs"])
        for (regime, dv), (xs, stats) in aggregated.items():
            for field, (mean, std, count) in stats.items():
                for i, x in enumerate(xs):
                    if count[i]:
                        w.writerow([
                            regime, f"{dv:.3f}", x, field,
                            mean[i], std[i], count[i]
                        ])


def plot_mean_band(ax, xs, stats, field, label=None):
    if field not in stats:
        return False
    mean, std, _ = stats[field]
    ax.plot(xs, mean, label=label or field)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.18)
    return True


def make_redistribute_plot(regime, dv, xs, stats, out_path):
    """
    Four-panel plot:
      1. relative redistributed mass per recorded interval
      2. inferred local/global redistribution events per interval
      3. projection violations / needs_projection
      4. net mass change and pre-projection extrema
    """
    if len(xs) == 0:
        return False

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1. Relative mass redistributed per history interval.
    ax = axes[0]
    plotted = False
    for field, label in [
        ("rel_d_mass_redistributed_low_local", "low local"),
        ("rel_d_mass_redistributed_low_global", "low global"),
        ("rel_d_mass_redistributed_simplex_local", "simplex local"),
        ("rel_d_mass_redistributed_simplex_global", "simplex global"),
        ("rel_d_mass_redistributed_high_local", "high local"),
        ("rel_d_mass_redistributed_high_global", "high global"),
    ]:
        plotted |= plot_mean_band(ax, xs, stats, field, label)
    ax.set_ylabel("Redistributed mass / reference mass")
    ax.set_title("Redistributed mass per recorded interval")
    if plotted:
        ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.25)

    # 2. Local vs global redistribution events.
    ax = axes[1]
    plotted = False
    for field, label in [
        ("event_low_local", "low local"),
        ("event_low_global", "low global"),
        ("event_simplex_local", "simplex local"),
        ("event_simplex_global", "simplex global"),
        ("event_high_local", "high local"),
        ("event_high_global", "high global"),
    ]:
        plotted |= plot_mean_band(ax, xs, stats, field, label)
    ax.set_ylabel("Fraction of runs with event")
    ax.set_title("Redistribution activity")
    if plotted:
        ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.25)

    # 3. Raw projection violations from this record.
    ax = axes[2]
    plotted = False
    for field, label in [
        ("n_low_entries", "low entries"),
        ("n_simplex_cells", "simplex cells"),
        ("n_high_entries", "high entries"),
    ]:
        plotted |= plot_mean_band(ax, xs, stats, field, label)
    ax.set_ylabel("Count")
    ax.set_title("Projection violations per recorded interval")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 4. Net projection mass drift. This is the conservation sanity check.
    ax = axes[3]
    if "net_change" in stats:
        plot_mean_band(ax, xs, stats, "net_change", "net mass change")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("Mass change")
    ax.set_xlabel("Simulation step")
    ax.set_title("Projection mass-conservation check")
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"Redistribute projection diagnostics: {regime}, D_v={dv:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_per_run(path, rows, numeric_fields_order):
    meta_fields = [
        "projection_mode", "projection_floor", "projection_tol",
        "dt", "nsteps", "frames", "noise", "runtime_seconds",
    ]
    header = [
        "regime", "D_v", "n_run", "diag_path", "history_path", "history_records",
    ] + meta_fields + numeric_fields_order

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_by_parameter(path, grouped, numeric_fields_order):
    header = ["regime", "D_v", "n_runs"]
    for field in numeric_fields_order:
        header += [f"{field}_mean", f"{field}_std"]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for (regime, dv), rows in grouped.items():
            out = {"regime": regime, "D_v": f"{dv:.3f}", "n_runs": len(rows)}
            for field in numeric_fields_order:
                vals = [
                    as_float(r.get(field))
                    for r in rows
                    if as_float(r.get(field)) is not None
                ]
                if vals:
                    vals = np.asarray(vals, dtype=float)
                    out[f"{field}_mean"] = vals.mean()
                    out[f"{field}_std"] = vals.std(ddof=0)
                else:
                    out[f"{field}_mean"] = np.nan
                    out[f"{field}_std"] = np.nan
            w.writerow(out)


def build_parser():
    p = argparse.ArgumentParser(
        description="Summarize clip/redistribute projection diagnostics."
    )
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--input-pattern",
        default="VS_run_{n_run}_det_Dv{D_v:.3f}",
        help=(
            "Base filename pattern without _projection_diag.json. "
            "Fields: {regime}, {n_run}/{run}, {D_v}/{Dv}/{dv}"
        ),
    )
    p.add_argument("--output-dir", default="projection_summary")
    p.add_argument("--strict", action="store_true")
    return p


def main(args):
    root = Path(args.data_root)
    outdir = Path(args.output_dir)
    plotdir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    grouped = defaultdict(list)
    grouped_histories = defaultdict(list)
    missing = []
    all_numeric = set()

    for regime in REGIMES:
        for dv in DV_VALS:
            for n_run in range(N_RUNS):
                base = root / regime / render_pattern(
                    args.input_pattern, regime, n_run, float(dv)
                )
                jpath = Path(f"{base}_projection_diag.json")
                hpath = Path(f"{base}_projection_history.csv")

                if not jpath.exists():
                    missing.append([regime, f"{dv:.3f}", n_run, str(jpath), "missing JSON"])
                    if args.strict:
                        raise FileNotFoundError(jpath)
                    continue

                try:
                    metadata, summary = read_json_diag(jpath)
                    history = read_history(hpath) if hpath.exists() else []

                    ref_mass = infer_reference_mass(metadata, summary, history)
                    history = add_history_derived_fields(history, ref_mass)
                    summary2, ref_mass = add_summary_derived(
                        summary, metadata, history
                    )

                    row = {
                        "regime": regime,
                        "D_v": float(dv),
                        "n_run": n_run,
                        "diag_path": str(jpath),
                        "history_path": str(hpath) if hpath.exists() else "",
                        "history_records": len(history),
                    }

                    for k in [
                        "projection_mode", "projection_floor", "projection_tol",
                        "dt", "nsteps", "frames", "noise", "runtime_seconds",
                    ]:
                        if k in metadata:
                            row[k] = metadata[k]

                    for k, v in summary2.items():
                        fv = as_float(v)
                        if fv is not None:
                            row[k] = fv
                            all_numeric.add(k)

                    run_rows.append(row)
                    grouped[(regime, float(dv))].append(row)

                    if history:
                        grouped_histories[(regime, float(dv))].append(history)

                except Exception as exc:
                    missing.append([
                        regime, f"{dv:.3f}", n_run, str(jpath),
                        f"{type(exc).__name__}: {exc}"
                    ])
                    print(f"Failed {jpath}: {exc}", file=sys.stderr)
                    if args.strict:
                        raise

    preferred = [
        "reference_mass",
        "net_mass_change_projection",
        "relative_mass_redistributed_total",
        "relative_mass_redistributed_local_total",
        "relative_mass_redistributed_global_total",
        "global_redistributed_mass_fraction",
        "relative_mass_redistributed_low_local",
        "relative_mass_redistributed_low_global",
        "relative_mass_redistributed_simplex_local",
        "relative_mass_redistributed_simplex_global",
        "relative_mass_redistributed_high_local",
        "relative_mass_redistributed_high_global",
        "n_redistribute_low_local_events",
        "n_redistribute_low_global_events",
        "n_redistribute_simplex_local_events",
        "n_redistribute_simplex_global_events",
        "n_redistribute_high_local_events",
        "n_redistribute_high_global_events",
        "n_redistribute_local_events_total",
        "n_redistribute_global_events_total",
        "global_fallback_event_fraction",
        "n_redistribute_low_failures",
        "n_redistribute_high_failures",
        "n_redistribute_simplex_failures",
        "n_redistribute_failures_total",
        "global_failure_per_global_event",
        *REDIST_LEFTOVER_FIELDS,
    ]
    ordered = [f for f in preferred if f in all_numeric]
    ordered += sorted(f for f in all_numeric if f not in ordered)

    run_csv = outdir / "projection_per_run.csv"
    param_csv = outdir / "projection_by_parameter.csv"
    hist_csv = outdir / "projection_history_by_parameter.csv"

    write_per_run(run_csv, run_rows, ordered)
    write_by_parameter(param_csv, grouped, ordered)

    aggregated = {}
    for key, histories in grouped_histories.items():
        aggregated[key] = aggregate_history(histories)

    write_long_history(hist_csv, aggregated)

    nplots = 0
    for (regime, dv), (xs, stats) in aggregated.items():
        p = plotdir / f"{regime}_Dv{dv:.3f}_redistribute_projection.png"
        if make_redistribute_plot(regime, dv, xs, stats, p):
            nplots += 1

    if missing:
        miss_csv = outdir / "missing_projection_diagnostics.csv"
        with miss_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["regime", "D_v", "n_run", "path", "status"])
            w.writerows(missing)

    print("Done.")
    print(f"Runs summarized: {len(run_rows)}")
    print(f"Plots written: {nplots}")
    print(run_csv)
    print(param_csv)
    print(hist_csv)
    print(plotdir)

    # Headline mass-conservation check.
    drift = np.asarray([
        as_float(r.get("net_mass_change_projection"))
        for r in run_rows
        if as_float(r.get("net_mass_change_projection")) is not None
    ], dtype=float)
    if len(drift):
        print("\nProjection mass-conservation check:")
        print(f"mean signed drift/run = {drift.mean():+.6e}")
        print(f"mean |drift|/run      = {np.abs(drift).mean():.6e}")
        print(f"max  |drift|/run      = {np.abs(drift).max():.6e}")


if __name__ == "__main__":
    main(build_parser().parse_args())
