#!/usr/bin/env python3
"""Summarize saved projection diagnostics for a Schelling-Voter sweep."""

from __future__ import annotations
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REGIMES = ["segregating", "integrating", "migrating", "well-mixed"]
DV_VALS = np.linspace(0.01, 0.2, 20)
N_RUNS = 10

KEY_SUMMARY_FIELDS = [
    "net_mass_change_projection", "expected_net_change", "bookkeeping_error",
    "mass_added_low", "mass_removed_high", "mass_removed_simplex",
    "mass_added_transfer_fallback", "mass_roundoff_cleanup",
]
X_CANDIDATES = ["snapshot", "frame", "record", "history_index", "step", "n", "iteration", "timestep", "time"]
KEY_HISTORY_FIELDS = [
    "net_mass_change_projection", "mass_added_low", "mass_removed_high",
    "mass_removed_simplex", "mass_added_transfer_fallback", "mass_roundoff_cleanup",
    "n_low", "n_high", "n_simplex", "num_low", "num_high", "num_simplex",
    "min_before_projection", "max_before_projection", "max_sum_before_projection",
]


def render_pattern(pattern, regime, n_run, dv):
    return pattern.format(regime=regime, n_run=n_run, run=n_run, D_v=dv, Dv=dv, dv=dv)


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


def compute_bookkeeping(summary):
    out = dict(summary)
    expected = (
        float(out.get("mass_added_low", 0.0) or 0.0)
        - float(out.get("mass_removed_high", 0.0) or 0.0)
        - float(out.get("mass_removed_simplex", 0.0) or 0.0)
        + float(out.get("mass_added_transfer_fallback", 0.0) or 0.0)
        + float(out.get("mass_roundoff_cleanup", 0.0) or 0.0)
    )
    out.setdefault("expected_net_change", expected)
    net, exp = as_float(out.get("net_mass_change_projection")), as_float(out.get("expected_net_change"))
    if net is not None and exp is not None:
        out.setdefault("bookkeeping_error", net - exp)
    return out


def read_json_diag(path):
    with path.open("r") as f:
        obj = json.load(f)
    return obj.get("metadata", {}), compute_bookkeeping(obj.get("projection_summary", {}))


def read_history_csv(path):
    rows = []
    with path.open("r", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            out = {"_row_index": float(i)}
            for k, v in row.items():
                if v in (None, ""):
                    continue
                fv = as_float(v)
                out[k] = fv if fv is not None else v
            rows.append(out)
    return rows


def choose_x_field(rows):
    if not rows:
        return "_row_index"
    keys = set().union(*(r.keys() for r in rows))
    for k in X_CANDIDATES:
        if k in keys:
            return k
    return "_row_index"


def numeric_fields(rows, exclude=()):
    fields, exclude = set(), set(exclude)
    for r in rows:
        for k, v in r.items():
            if k not in exclude and not k.startswith("_") and as_float(v) is not None:
                fields.add(k)
    return sorted(fields)


def aggregate_history(run_histories):
    if not run_histories:
        return None, np.array([]), {}
    chosen = [choose_x_field(rows) for rows in run_histories if rows]
    if not chosen:
        return None, np.array([]), {}
    x_name = chosen[0] if all(x == chosen[0] for x in chosen) else "_row_index"
    all_fields = set()
    for rows in run_histories:
        all_fields.update(numeric_fields(rows, exclude={x_name}))
    bucket = {field: defaultdict(list) for field in sorted(all_fields)}
    x_values = set()
    for rows in run_histories:
        for j, row in enumerate(rows):
            x = as_float(row.get(x_name, j))
            if x is None:
                x = float(j)
            x_values.add(x)
            for field in bucket:
                v = as_float(row.get(field))
                if v is not None:
                    bucket[field][x].append(v)
    xs = np.array(sorted(x_values), dtype=float)
    stats = {}
    for field in bucket:
        means = np.full(len(xs), np.nan)
        stds = np.full(len(xs), np.nan)
        counts = np.zeros(len(xs), dtype=int)
        for i, x in enumerate(xs):
            vals = bucket[field].get(x, [])
            if vals:
                vals = np.asarray(vals, dtype=float)
                means[i], stds[i], counts[i] = vals.mean(), vals.std(ddof=0), len(vals)
        stats[field] = (means, stds, counts)
    return x_name, xs, stats


def prioritize_fields(fields):
    fields = list(fields)
    ordered = [f for f in KEY_HISTORY_FIELDS if f in fields]
    return ordered + [f for f in fields if f not in ordered]


def plot_parameter_history(regime, dv, x_name, xs, stats, out_path, max_panels=6):
    if len(xs) == 0 or not stats:
        return False
    fields = prioritize_fields(stats.keys())
    informative = []
    for field in fields:
        mean, std, _ = stats[field]
        finite = mean[np.isfinite(mean)]
        if len(finite) and (np.any(np.abs(finite) > 0) or np.any(np.nan_to_num(std) > 0)):
            informative.append(field)
    if not informative:
        informative = fields[:max_panels]
    else:
        informative = informative[:max_panels]
    if not informative:
        return False
    n = len(informative)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(3.0, 2.35*n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, field in zip(axes, informative):
        mean, std, _ = stats[field]
        ax.plot(xs, mean, label="mean over runs")
        ax.fill_between(xs, mean-std, mean+std, alpha=0.2, label="±1 std")
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylabel(field)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel(x_name)
    fig.suptitle(f"Projection diagnostics: {regime}, D_v={dv:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_run_summary(path, rows, summary_fields):
    metadata_fields = ["projection_mode", "projection_floor", "projection_tol", "dt", "nsteps", "frames", "noise", "runtime_seconds"]
    header = ["regime", "D_v", "n_run", "diag_path", "history_path", "history_records"] + metadata_fields + summary_fields
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_parameter_summary(path, grouped_rows, fields):
    header = ["regime", "D_v", "n_runs"]
    for field in fields:
        header += [f"{field}_mean", f"{field}_std", f"{field}_min", f"{field}_max"]
    header += ["mean_abs_net_mass_change_projection", "max_abs_net_mass_change_projection", "fraction_runs_nonzero_net_drift"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for (regime, dv), rows in sorted(grouped_rows.items()):
            out = {"regime": regime, "D_v": f"{dv:.3f}", "n_runs": len(rows)}
            for field in fields:
                vals = np.asarray([as_float(r.get(field)) for r in rows if as_float(r.get(field)) is not None], dtype=float)
                if len(vals):
                    out[f"{field}_mean"], out[f"{field}_std"] = vals.mean(), vals.std(ddof=0)
                    out[f"{field}_min"], out[f"{field}_max"] = vals.min(), vals.max()
                else:
                    for suffix in ("mean", "std", "min", "max"):
                        out[f"{field}_{suffix}"] = np.nan
            drift = np.asarray([as_float(r.get("net_mass_change_projection")) for r in rows if as_float(r.get("net_mass_change_projection")) is not None], dtype=float)
            if len(drift):
                out["mean_abs_net_mass_change_projection"] = np.mean(np.abs(drift))
                out["max_abs_net_mass_change_projection"] = np.max(np.abs(drift))
                out["fraction_runs_nonzero_net_drift"] = np.mean(np.abs(drift) > 1e-14)
            else:
                out["mean_abs_net_mass_change_projection"] = np.nan
                out["max_abs_net_mass_change_projection"] = np.nan
                out["fraction_runs_nonzero_net_drift"] = np.nan
            w.writerow(out)


def write_history_summary(path, aggregated):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "D_v", "x_name", "x", "diagnostic", "mean", "std", "n_runs"])
        for (regime, dv), (x_name, xs, stats) in sorted(aggregated.items()):
            for field, (means, stds, counts) in stats.items():
                for i, x in enumerate(xs):
                    if counts[i]:
                        w.writerow([regime, f"{dv:.3f}", x_name, x, field, means[i], stds[i], counts[i]])


def build_parser():
    p = argparse.ArgumentParser(description="Summarize projection diagnostics for a parameter sweep.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--input-pattern", default="VS_run_{n_run}_det_Dv{D_v:.3f}", help="Run base-name pattern without diagnostic suffix.")
    p.add_argument("--output-dir", default="projection_summary")
    p.add_argument("--max-plot-panels", type=int, default=6)
    p.add_argument("--strict", action="store_true")
    return p


def main(args):
    data_root, out_dir = Path(args.data_root), Path(args.output_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_rows, grouped_rows, grouped_histories, missing = [], defaultdict(list), defaultdict(list), []
    all_fields = set()
    expected, scanned = len(REGIMES)*len(DV_VALS)*N_RUNS, 0

    for regime in REGIMES:
        for dv in DV_VALS:
            for n_run in range(N_RUNS):
                base = data_root / regime / render_pattern(args.input_pattern, regime, n_run, float(dv))
                jp = Path(f"{base}_projection_diag.json")
                hp = Path(f"{base}_projection_history.csv")
                scanned += 1
                if not jp.exists():
                    missing.append([regime, f"{dv:.3f}", n_run, str(jp), "missing JSON"])
                    if args.strict:
                        raise FileNotFoundError(jp)
                    continue
                try:
                    metadata, summary = read_json_diag(jp)
                    numeric = {k: as_float(v) for k, v in summary.items() if as_float(v) is not None}
                    all_fields.update(numeric)
                    history = read_history_csv(hp) if hp.exists() else []
                    if history:
                        grouped_histories[(regime, float(dv))].append(history)
                    row = {
                        "regime": regime, "D_v": float(dv), "n_run": n_run,
                        "diag_path": str(jp), "history_path": str(hp) if hp.exists() else "",
                        "history_records": len(history),
                    }
                    for field in ["projection_mode", "projection_floor", "projection_tol", "dt", "nsteps", "frames", "noise", "runtime_seconds"]:
                        if field in metadata:
                            row[field] = metadata[field]
                    row.update(numeric)
                    run_rows.append(row)
                    grouped_rows[(regime, float(dv))].append(row)
                except Exception as exc:
                    missing.append([regime, f"{dv:.3f}", n_run, str(jp), f"{type(exc).__name__}: {exc}"])
                    print(f"Failed {jp}: {exc}", file=sys.stderr)
                    if args.strict:
                        raise
                if scanned % 50 == 0:
                    print(f"Scanned {scanned}/{expected} expected runs")

    fields = [f for f in KEY_SUMMARY_FIELDS if f in all_fields] + sorted(f for f in all_fields if f not in KEY_SUMMARY_FIELDS)
    run_csv = out_dir / "projection_per_run.csv"
    param_csv = out_dir / "projection_by_parameter.csv"
    hist_csv = out_dir / "projection_history_by_parameter.csv"
    write_run_summary(run_csv, run_rows, fields)
    write_parameter_summary(param_csv, grouped_rows, fields)

    aggregated = {key: aggregate_history(histories) for key, histories in grouped_histories.items()}
    write_history_summary(hist_csv, aggregated)
    nplots = 0
    for (regime, dv), (x_name, xs, stats) in aggregated.items():
        if x_name is None:
            continue
        nplots += int(plot_parameter_history(regime, dv, x_name, xs, stats, plot_dir / f"{regime}_Dv{dv:.3f}_projection_history.png", args.max_plot_panels))

    if missing:
        mp = out_dir / "missing_projection_diagnostics.csv"
        with mp.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["regime", "D_v", "n_run", "path", "status"]); w.writerows(missing)
        print(f"Missing/failed diagnostics: {len(missing)} -> {mp}")

    drift = np.asarray([as_float(r.get("net_mass_change_projection")) for r in run_rows if as_float(r.get("net_mass_change_projection")) is not None], dtype=float)
    print("\nDone.")
    print(f"Successful JSON diagnostics: {len(run_rows)}")
    print(f"Parameter settings with history: {len(aggregated)}")
    print(f"Plots written: {nplots}")
    print(f"  {run_csv}\n  {param_csv}\n  {hist_csv}\n  {plot_dir}")
    if len(drift):
        print("\nOverall projection mass-drift check:")
        print(f"  mean signed drift/run = {drift.mean():+.6e}")
        print(f"  mean |drift|/run      = {np.abs(drift).mean():.6e}")
        print(f"  max  |drift|/run      = {np.abs(drift).max():.6e}")
        print(f"  runs with |drift| > 1e-14 = {np.count_nonzero(np.abs(drift) > 1e-14)}/{len(drift)}")

if __name__ == "__main__":
    main(build_parser().parse_args())
