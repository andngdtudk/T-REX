#!/usr/bin/env python3
"""Export each model's BEST performance per metric (peak across all epochs,
not just a fixed final epoch), as a CSV with metrics as rows and models as
columns. Each cell is "mean \\pm 95%-CI", e.g. "455.12 \\pm 2.23".

"Best" means: smallest for metrics where lower is better (duration,
waiting time, never-arrived, queue length, time loss), largest where higher
is better (throughput).

Reuses every parsing/aggregation function from plot_results.py, so the
underlying metrics always match what's plotted. Run this from the same
directory as plot_results.py (it imports it), with the same
results/<test>/<run>/ layout underneath.

How the "best epoch" is chosen: for each model, the across-run mean at each
epoch is first rolling-averaged (same ROLLING_WINDOW used by the plots) to
avoid picking a single noisy fluke epoch as "best". Whichever epoch is best
on that smoothed curve is used, but the reported mean/CI is then computed
from the RAW (unsmoothed) per-run values at that epoch, not the smoothed
value -- so the number shown is a real, honest average of actual runs.
"""
import os
import csv
import math
import statistics

import plot5 as pr

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


OUTPUT_CSV = "best_performance.csv"
BEST_EPOCHS_CSV = "best_performance_epochs.csv"  # secondary/informational
DECIMALS = 2


# ---------------------------------------------------------------------------
# Stats helpers (same conventions as export_summary_csv.py)
# ---------------------------------------------------------------------------

def t_multiplier(n):
    """95% two-sided critical value for `n` samples (n-1 degrees of
    freedom). Falls back to the large-sample normal-approximation value
    (1.96) if scipy isn't installed."""
    if n < 2:
        return math.nan
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(0.975, df=n - 1))
    print(
        "Warning: scipy not installed -- using the 1.96 normal-approximation "
        "critical value instead of the exact t-distribution value. Run "
        "`pip install scipy` for an exact 95% CI (matters most when you "
        "only have a few runs per test)."
    )
    return 1.96


def mean_ci95(values):
    """values: list of floats (NaN/None are filtered out here). Returns
    (mean, ci95_halfwidth, n)."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return math.nan, math.nan, 0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, n
    sem = statistics.stdev(vals) / math.sqrt(n)  # sample stdev (n-1 denom)
    return m, t_multiplier(n) * sem, n


def fmt(mean, ci95):
    if isinstance(mean, float) and math.isnan(mean):
        return ""
    return f"{mean:.{DECIMALS}f} \\pm {ci95:.{DECIMALS}f}"


# ---------------------------------------------------------------------------
# Best-epoch selection
# ---------------------------------------------------------------------------

def _raw_value(entry, subkey):
    if subkey is None:
        return entry
    return entry.get(subkey, math.nan) if isinstance(entry, dict) else math.nan


def best_epoch_stats(per_run_series, direction, subkey=None, smooth_window=pr.ROLLING_WINDOW):
    """per_run_series: {test: {run: {epoch: value_or_dict}}}.

    For every test: builds the raw (unsmoothed) across-run mean at each
    epoch, rolling-averages that curve, and finds the epoch that's best
    (min/max per `direction`) on the smoothed curve. The value reported for
    that test is then the mean+CI95 of the RAW per-run values at that same
    epoch.

    Returns {test: (mean, ci95, n, best_epoch)}.
    """
    assert direction in ("min", "max")
    result = {}

    for test, runs in per_run_series.items():
        all_epochs = set()
        for epochs in runs.values():
            all_epochs.update(epochs.keys())

        if not all_epochs:
            result[test] = (math.nan, math.nan, 0, None)
            continue

        sorted_epochs = sorted(all_epochs)
        raw_means = []
        for e in sorted_epochs:
            vals = [
                _raw_value(runs[run][e], subkey)
                for run in runs if e in runs[run]
            ]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            raw_means.append(sum(vals) / len(vals) if vals else math.nan)

        smoothed = pr.rolling_average(raw_means, smooth_window)

        candidates = [
            (v, e) for v, e in zip(smoothed, sorted_epochs)
            if not (isinstance(v, float) and math.isnan(v))
        ]
        if not candidates:
            result[test] = (math.nan, math.nan, 0, None)
            continue

        pick = min if direction == "min" else max
        _best_val, best_epoch = pick(candidates, key=lambda t: t[0])

        raw_vals_at_best = [
            _raw_value(runs[run][best_epoch], subkey)
            for run in runs if best_epoch in runs[run]
        ]
        mean, ci95, n = mean_ci95(raw_vals_at_best)
        result[test] = (mean, ci95, n, best_epoch)

    return result


def per_run_mode_series(per_run_wait_stats, mode):
    """per_run_wait_stats: {test: {run: {mode: {epoch: stat_dict}}}}.
    Returns just one mode's view: {test: {run: {epoch: stat_dict}}}."""
    result = {}
    for test, runs in per_run_wait_stats.items():
        result[test] = {run: per_mode.get(mode, {}) for run, per_mode in runs.items()}
    return result


# ---------------------------------------------------------------------------
# Build the best-performance table
# ---------------------------------------------------------------------------

def build_best_performance(results_dir):
    raw_records = pr.collect_raw_records(results_dir)
    never_arrived = pr.collect_never_arrived(results_dir)
    episode_metrics = pr.collect_episode_metrics(results_dir)

    all_tests = sorted(set(raw_records) | set(never_arrived) | set(episode_metrics))

    # Each row: (row_label, direction, per_run_series, subkey)
    row_specs = []

    if raw_records:
        per_run_combined = pr.compute_per_run(raw_records, pr.compute_combined_metric_averages)
        per_run_throughput = pr.compute_per_run(raw_records, pr.compute_combined_throughput)
        per_run_wait_stats = pr.compute_per_run(raw_records, pr.compute_per_mode_wait_stats)

        row_specs.append(("duration", "min", per_run_combined, "duration"))

        for mode in pr.ALL_MODES:
            mode_series = per_run_mode_series(per_run_wait_stats, mode)
            row_specs.append((f"waitingTime_{mode}", "min", mode_series, "average"))

        row_specs.append(("waitingTime_combined", "min", per_run_combined, "waitingTime"))

        row_specs.append(("timeLoss", "min", per_run_combined, "timeLoss"))

        for mode in pr.ALL_MODES:
            mode_series = per_run_mode_series(per_run_wait_stats, mode)
            row_specs.append((f"throughput_{mode}", "max", mode_series, "throughput"))

        row_specs.append(("throughput_combined", "max", per_run_throughput, None))

    if never_arrived:
        for mode in pr.ALL_MODES:
            row_specs.append((f"never_arrived_{mode}", "min", never_arrived, mode))

    if episode_metrics:
        queue_metrics = pr.compute_queue_metrics_from_episode(episode_metrics)
        for key in ("car", "bike", "ped"):
            row_specs.append((f"queue_{key}", "min", queue_metrics, key))
        row_specs.append(("queue_combined", "min", queue_metrics, "combined"))

    # row_label -> {test: (mean, ci95, n, best_epoch)}
    rows = {}
    for row_label, direction, per_run_series, subkey in row_specs:
        rows[row_label] = best_epoch_stats(per_run_series, direction, subkey=subkey)

    return all_tests, row_specs, rows


def write_csvs(all_tests, row_specs, rows, output_path, epochs_output_path):
    row_labels = [label for label, *_ in row_specs]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + all_tests)
        for label in row_labels:
            per_test = rows[label]
            row = [label]
            for test in all_tests:
                mean, ci95, _n, _epoch = per_test.get(test, (math.nan, math.nan, 0, None))
                row.append(fmt(mean, ci95))
            writer.writerow(row)

    # Secondary/informational: which epoch each "best" value came from, for
    # reproducibility -- not required by the requested format, but cheap to
    # also have on hand.
    with open(epochs_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + all_tests)
        for label in row_labels:
            per_test = rows[label]
            row = [label]
            for test in all_tests:
                _mean, _ci95, _n, epoch = per_test.get(test, (math.nan, math.nan, 0, None))
                row.append("" if epoch is None else epoch)
            writer.writerow(row)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, pr.RESULTS_DIR)
    output_path = os.path.join(base_dir, OUTPUT_CSV)
    epochs_output_path = os.path.join(base_dir, BEST_EPOCHS_CSV)

    all_tests, row_specs, rows = build_best_performance(results_dir)

    if not all_tests:
        print(f"No results found under {results_dir}/<test>/<run>/.")
        return

    write_csvs(all_tests, row_specs, rows, output_path, epochs_output_path)
    print(f"Wrote best-performance table ({len(row_specs)} metrics x {len(all_tests)} models) to {output_path}")
    print(f"Wrote which epoch each 'best' value came from to {epochs_output_path}")


if __name__ == "__main__":
    main()