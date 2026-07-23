#!/usr/bin/env python3
"""Export a per-model summary table (mean +/- 95% CI) at a fixed "final"
epoch, as a CSV -- one row per test/model, one column per metric, with each
cell formatted as e.g. "455.12 \\pm 2.23".

Reuses every parsing/aggregation function from plot5.py, so the
metrics reported here always match what's plotted -- this script just reads
the same per-run series and reports a single epoch's cross-run mean and 95%
confidence interval instead of a full curve.

Run this from the same directory as plot5.py (it imports it), with
the same results/<test>/<run>/ layout underneath.
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


# The epoch/episode reported for ordinary tests.
FINAL_EPOCH = 100
# The epoch/episode reported for long-horizon tests (pr.MARKER_PREFIXES,
# i.e. IPPO/FMA2C), which are trained for far longer.
FINAL_EPOCH_LONG = 1400

OUTPUT_CSV = "summary_at_final_epoch.csv"
DECIMALS = 2


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def is_long_horizon(test_name):
    return test_name.upper().startswith(pr.MARKER_PREFIXES)


def final_epoch_for(test_name):
    return FINAL_EPOCH_LONG if is_long_horizon(test_name) else FINAL_EPOCH


def t_multiplier(n):
    """95% two-sided critical value for `n` samples (n-1 degrees of
    freedom). Falls back to the large-sample normal-approximation value
    (1.96) if scipy isn't installed -- install scipy for an exact value
    when n is small (as it will be with only a handful of runs)."""
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
    """values: list of floats (NaN/None already expected to be filtered out
    by the caller). Returns (mean, ci95_halfwidth, n)."""
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
# Epoch lookup -- tolerant of the data not landing on exactly epoch 100/1400
# (e.g. tripinfo files spaced every 2 or 5 epochs)
# ---------------------------------------------------------------------------

def _nearest_entry(epoch_dict, target_epoch):
    """epoch_dict: {epoch: value}. Returns (actual_epoch_used, value) for
    whichever epoch is closest to target_epoch (ties broken toward the
    smaller epoch). Returns (None, None) if epoch_dict is empty."""
    if not epoch_dict:
        return None, None
    closest = min(epoch_dict.keys(), key=lambda e: (abs(e - target_epoch), e))
    return closest, epoch_dict[closest]


def values_at_final_epoch(per_run_series, metric_label):
    """per_run_series: {test: {run: {epoch: scalar_value}}}.

    For every test, looks up (or finds the nearest available epoch to)
    that test's target final epoch in every run. Returns {test: [values]}.
    """
    result = {}
    for test, runs in per_run_series.items():
        target = final_epoch_for(test)
        vals = []
        used_epochs = set()
        for _run, epochs in runs.items():
            actual_epoch, value = _nearest_entry(epochs, target)
            if actual_epoch is None:
                continue
            used_epochs.add(actual_epoch)
            vals.append(value)
        if used_epochs and used_epochs != {target}:
            print(
                f"Note: {metric_label} for {test!r} -- target epoch {target} "
                f"wasn't present in every run; used nearest available "
                f"epoch(s) {sorted(used_epochs)} instead."
            )
        result[test] = vals
    return result


def subkey_values_at_final_epoch(per_run_series, subkey, metric_label):
    """Same as values_at_final_epoch, but for series where each epoch's
    entry is itself a dict, e.g. {test: {run: {epoch: {subkey: value}}}}."""
    result = {}
    for test, runs in per_run_series.items():
        target = final_epoch_for(test)
        vals = []
        used_epochs = set()
        for _run, epochs in runs.items():
            actual_epoch, row = _nearest_entry(epochs, target)
            if actual_epoch is None:
                continue
            used_epochs.add(actual_epoch)
            vals.append(row.get(subkey, math.nan))
        if used_epochs and used_epochs != {target}:
            print(
                f"Note: {metric_label} for {test!r} -- target epoch {target} "
                f"wasn't present in every run; used nearest available "
                f"epoch(s) {sorted(used_epochs)} instead."
            )
        result[test] = vals
    return result


def per_run_mode_series(per_run_wait_stats, mode):
    """per_run_wait_stats: {test: {run: {mode: {epoch: stat_dict}}}}.
    Returns just one mode's view: {test: {run: {epoch: stat_dict}}}."""
    result = {}
    for test, runs in per_run_wait_stats.items():
        result[test] = {run: per_mode.get(mode, {}) for run, per_mode in runs.items()}
    return result


# ---------------------------------------------------------------------------
# Build the summary table
# ---------------------------------------------------------------------------

def build_summary(results_dir):
    raw_records = pr.collect_raw_records(results_dir)
    never_arrived = pr.collect_never_arrived(results_dir)
    episode_metrics = pr.collect_episode_metrics(results_dir)

    all_tests = sorted(set(raw_records) | set(never_arrived) | set(episode_metrics))

    # column_name -> {test: formatted_string}
    columns = {}

    def add_column(column_name, per_run_series, metric_label, subkey=None):
        if subkey is None:
            per_test_values = values_at_final_epoch(per_run_series, metric_label)
        else:
            per_test_values = subkey_values_at_final_epoch(per_run_series, subkey, metric_label)

        formatted = {}
        for test in all_tests:
            mean, ci95, _n = mean_ci95(per_test_values.get(test, []))
            formatted[test] = fmt(mean, ci95)
        columns[column_name] = formatted

    # --- combined tripinfo/personinfo metrics (duration/waitingTime/timeLoss) ---
    if raw_records:
        per_run_combined = pr.compute_per_run(raw_records, pr.compute_combined_metric_averages)
        for metric in pr.METRICS:
            add_column(metric, per_run_combined, f"combined {metric}", subkey=metric)

        # --- combined throughput ---
        per_run_throughput = pr.compute_per_run(raw_records, pr.compute_combined_throughput)
        add_column("throughput_combined", per_run_throughput, "combined throughput")

        # --- per-mode wait stats ---
        per_run_wait_stats = pr.compute_per_run(raw_records, pr.compute_per_mode_wait_stats)
        for mode in pr.ALL_MODES:
            mode_series = per_run_mode_series(per_run_wait_stats, mode)
            for stat_key, col_stub in (
                ("total", "wait_total"),
                ("average", "wait_average"),
                ("p90", "wait_p90"),
                ("variance", "wait_variance"),
                ("throughput", "throughput"),
            ):
                add_column(f"{col_stub}_{mode}", mode_series, f"{mode} {stat_key} wait", subkey=stat_key)

    # --- never arrived ---
    if never_arrived:
        for mode in pr.ALL_MODES:
            add_column(f"never_arrived_{mode}", never_arrived, f"never arrived ({mode})", subkey=mode)

    # --- queue metrics (from mm_episode_metrics.csv) ---
    if episode_metrics:
        queue_metrics = pr.compute_queue_metrics_from_episode(episode_metrics)
        for key in ("car", "bike", "ped", "combined"):
            add_column(f"queue_{key}", queue_metrics, f"{key} queue", subkey=key)

        # --- other episode-level columns (reward, epsilon, mean_q, ...) ---
        for column, filename_stub, _ylabel, _title in pr.EPISODE_METRIC_SPECS:
            add_column(f"episode_{filename_stub}", episode_metrics, column, subkey=column)

    return all_tests, columns


def write_csv(all_tests, columns, output_path):
    column_names = list(columns.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + column_names)
        for test in all_tests:
            row = [test] + [columns[col].get(test, "") for col in column_names]
            writer.writerow(row)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, pr.RESULTS_DIR)
    output_path = os.path.join(base_dir, OUTPUT_CSV)

    all_tests, columns = build_summary(results_dir)

    if not all_tests:
        print(f"No results found under {results_dir}/<test>/<run>/.")
        return

    write_csv(all_tests, columns, output_path)
    print(f"Wrote summary for {len(all_tests)} models ({len(columns)} metrics) to {output_path}")


if __name__ == "__main__":
    main()