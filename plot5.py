import os
import re
import csv
import math
import statistics
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOTS_DIR = "plots"
TRIPINFO_PATTERN = re.compile(r"^tripinfo_(\d+)\.xml$")
PERSONINFO_PATTERN = re.compile(r"^personinfo_(\d+)\.xml$")

# One fixed-name step-metrics CSV per run (not per-epoch like tripinfo/personinfo).
# Expected columns include at least: episode, decision, car_queue, bike_queue.
# There is no pedestrian queue column, so queue plots only cover car + bike.
MM_STEP_METRICS_FILENAME = "mm_step_metrics.csv"

# Expected directory layout:
#   results/<test_name>/<run_name>/tripinfo_<epoch>.xml
#   results/<test_name>/<run_name>/personinfo_<epoch>.xml
#
# Each <test_name> is a method/configuration you want to compare; each
# <run_name> underneath it is an independent stochastic repetition of that
# same test. Plots show the mean across runs per epoch with a shaded error
# band, one line+band per test.

# Combined metrics plotted directly from tripinfo/personinfo attributes.
# stopTime is intentionally excluded (not plotted per latest instructions).
METRICS = ("duration", "waitingTime", "timeLoss")
PED_METRICS = {"duration", "waitingTime", "timeLoss"}  # fields available on personinfo

# If True, skip trips that are not finished (arrival == -1 or vaporized == "end").
SKIP_UNFINISHED = False

# Window (in seconds) used as the denominator for throughput calculations.
# throughput = (# arrivals in epoch) / (EPOCH_DURATION_SECONDS / 3600)
EPOCH_DURATION_SECONDS = 3600

# Window size (in epochs) for rolling-average smoothing plots.
ROLLING_WINDOW = 5

# Error band shown around the across-run mean.
#   "std" -> population standard deviation across runs (spread of runs)
#   "sem" -> standard error of the mean (std / sqrt(n), uncertainty on the mean)
ERROR_BAND_KIND = "std"

MODE_CAR = "car"
MODE_BIKE = "bike"
MODE_PED = "pedestrian"
ALL_MODES = (MODE_CAR, MODE_BIKE, MODE_PED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(raw):
    """Return float or None on failure."""
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _safe_int(raw):
    """Return int or None on failure (tolerates floats like '3.0')."""
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def rolling_average(values, window):
    """Simple trailing rolling average. NaNs are ignored within the window.

    Returns a list the same length as `values`. Early points use whatever
    history is available (a growing window) rather than NaN, so the line
    starts immediately instead of being truncated.
    """
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = [v for v in values[lo:i + 1] if not (isinstance(v, float) and math.isnan(v))]
        smoothed.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return smoothed


def percentile(values, pct):
    """Linear-interpolation percentile (matches numpy's default 'linear' method)."""
    if not values:
        return math.nan
    data = sorted(values)
    n = len(data)
    if n == 1:
        return data[0]
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def mean_and_error(values, kind=ERROR_BAND_KIND):
    """Mean and error (std or sem) across a list of values, ignoring NaN/None.

    Returns (mean, error, n_used). If nothing valid is present, returns
    (nan, nan, 0). If only one valid value is present, error is 0.
    """
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, len(vals)
    sd = statistics.pstdev(vals)
    if kind == "sem":
        sd = sd / math.sqrt(len(vals))
    return m, sd, len(vals)


def aggregate_scalar_series(per_run_epoch_scalar, kind=ERROR_BAND_KIND):
    """per_run_epoch_scalar: {run: {epoch: scalar}}

    Returns {epoch: (mean, error, n)} aggregated across runs. Epochs missing
    for a given run are treated as NaN (ignored) for that run.
    """
    all_epochs = set()
    for epochs in per_run_epoch_scalar.values():
        all_epochs.update(epochs.keys())

    aggregated = {}
    for epoch in all_epochs:
        values = [epochs.get(epoch, math.nan) for epochs in per_run_epoch_scalar.values()]
        aggregated[epoch] = mean_and_error(values, kind=kind)
    return aggregated


# ---------------------------------------------------------------------------
# Tripinfo parsing  (cars + bikes)
# ---------------------------------------------------------------------------

def iter_tripinfo_records(xml_path):
    """Yield (mode, metrics_dict) for each valid, finished tripinfo record.

    metrics_dict contains the raw float values for everything in METRICS,
    for that single trip.
    """
    records = []

    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag != "tripinfo":
                elem.clear()
                continue

            trip_id = elem.get("id", "")
            vtype = elem.get("vType", "")

            tokens = f"{trip_id} {vtype}".lower()
            if "bike" in tokens or "bicycle" in tokens or "cycle" in tokens:
                mode = MODE_BIKE
            else:
                mode = MODE_CAR

            if SKIP_UNFINISHED:
                arrival = elem.get("arrival")
                vaporized = elem.get("vaporized")
                if arrival == "-1" or vaporized == "end":
                    elem.clear()
                    continue

            values = {}
            valid = True
            for metric in METRICS:
                raw = elem.get(metric)
                value = _safe_float(raw)
                if value is None or value == -1:
                    valid = False
                    break
                values[metric] = value

            if valid:
                records.append((mode, values))

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return records


# ---------------------------------------------------------------------------
# Personinfo parsing  (pedestrians)
# ---------------------------------------------------------------------------

def iter_personinfo_records(xml_path):
    """Yield metrics_dict for each valid, finished personinfo record."""
    records = []

    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag != "personinfo":
                elem.clear()
                continue

            if SKIP_UNFINISHED:
                arrival_v = _safe_float(elem.get("arrival"))
                duration_v = _safe_float(elem.get("duration"))
                if (arrival_v is not None and arrival_v < 0) or \
                   (duration_v is not None and duration_v < 0):
                    elem.clear()
                    continue

            values = {}
            valid = True
            for metric in PED_METRICS:
                raw = elem.get(metric)
                value = _safe_float(raw)
                if value is None or value == -1:
                    valid = False
                    break
                values[metric] = value

            if valid:
                # METRICS may include fields not in PED_METRICS in the future;
                # fill any gap with 0.0 so downstream code stays uniform.
                full_values = {m: values.get(m, 0.0) for m in METRICS}
                records.append(full_values)

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return records


# ---------------------------------------------------------------------------
# Queue-length parsing (mm_step_metrics.csv, one fixed file per run)
# ---------------------------------------------------------------------------

def read_mm_step_metrics(csv_path):
    """Read a run's mm_step_metrics.csv.

    Returns a list of dicts: {"episode": int, "car_queue": float, "bike_queue": float}
    (queue values are math.nan when missing/unparsable). Rows without a
    usable episode number are skipped.
    """
    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                episode = _safe_int(raw_row.get("episode"))
                if episode is None:
                    continue

                car_queue = _safe_float(raw_row.get("car_queue"))
                bike_queue = _safe_float(raw_row.get("bike_queue"))

                rows.append({
                    "episode": episode,
                    "car_queue": car_queue if car_queue is not None else math.nan,
                    "bike_queue": bike_queue if bike_queue is not None else math.nan,
                })
    except (OSError, csv.Error) as exc:
        print(f"Warning: failed to read {csv_path}: {exc}")

    return rows


def compute_queue_averages_per_run(csv_rows):
    """Average car/bike queue length across decisions within each episode.

    csv_rows: list of {"episode", "car_queue", "bike_queue"} (one row per
    decision step within an episode).
    Returns: {episode: {"car": avg, "bike": avg, "combined": car_avg + bike_avg}}
    """
    by_episode = {}
    for row in csv_rows:
        bucket = by_episode.setdefault(row["episode"], {"car": [], "bike": []})
        bucket["car"].append(row["car_queue"])
        bucket["bike"].append(row["bike_queue"])

    result = {}
    for episode, vals in by_episode.items():
        car_vals = [v for v in vals["car"] if not (isinstance(v, float) and math.isnan(v))]
        bike_vals = [v for v in vals["bike"] if not (isinstance(v, float) and math.isnan(v))]

        car_avg = sum(car_vals) / len(car_vals) if car_vals else math.nan
        bike_avg = sum(bike_vals) / len(bike_vals) if bike_vals else math.nan

        if math.isnan(car_avg) and math.isnan(bike_avg):
            combined = math.nan
        else:
            combined = (0.0 if math.isnan(car_avg) else car_avg) + \
                       (0.0 if math.isnan(bike_avg) else bike_avg)

        result[episode] = {"car": car_avg, "bike": bike_avg, "combined": combined}

    return result


def collect_queue_metrics(results_dir):
    """Walk results_dir/<test>/<run>/mm_step_metrics.csv.

    Returns: {test: {run: {episode: {"car": avg, "bike": avg, "combined": avg}}}}
    Tests/runs without the CSV file (or with no usable rows) are simply
    omitted, so this feature degrades gracefully if some runs don't have it.
    """
    tests = {}

    if not os.path.isdir(results_dir):
        return tests

    for test_entry in sorted(os.listdir(results_dir)):
        test_path = os.path.join(results_dir, test_entry)
        if not os.path.isdir(test_path):
            continue

        runs = {}

        for run_entry in sorted(os.listdir(test_path)):
            run_path = os.path.join(test_path, run_entry)
            if not os.path.isdir(run_path):
                continue

            csv_path = os.path.join(run_path, MM_STEP_METRICS_FILENAME)
            if not os.path.isfile(csv_path):
                continue

            csv_rows = read_mm_step_metrics(csv_path)
            if not csv_rows:
                continue

            runs[run_entry] = compute_queue_averages_per_run(csv_rows)

        if runs:
            tests[test_entry] = runs

    return tests


def aggregate_queue_metrics(per_run_queue_metrics):
    """{test: {run: {episode: {"car"/"bike"/"combined": value}}}}
    -> {test: {"car"/"bike"/"combined": {episode: (mean, err, n)}}}
    """
    result = {}
    for test, runs in per_run_queue_metrics.items():
        result[test] = {}
        for key in ("car", "bike", "combined"):
            per_run_scalar = {
                run: {episode: vals[key] for episode, vals in episodes.items()}
                for run, episodes in runs.items()
            }
            result[test][key] = aggregate_scalar_series(per_run_scalar)
    return result


# ---------------------------------------------------------------------------
# Never-arrived helpers (unchanged behaviour)
# ---------------------------------------------------------------------------

def detect_vehicle_mode(trip_id, vtype):
    tokens = f"{trip_id or ''} {vtype or ''}".lower()
    if "bike" in tokens or "bicycle" in tokens or "cycle" in tokens:
        return MODE_BIKE
    return MODE_CAR


def iter_tripinfo_never_arrived(xml_path):
    counts = {MODE_CAR: 0, MODE_BIKE: 0}

    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag != "tripinfo":
                elem.clear()
                continue

            trip_id = elem.get("id", "")
            vtype = elem.get("vType", "")
            mode = detect_vehicle_mode(trip_id, vtype)

            duration_v = _safe_float(elem.get("duration"))
            arrival_v = _safe_float(elem.get("arrival"))

            if (duration_v is not None and duration_v < 0) or \
               (arrival_v is not None and arrival_v < 0):
                counts[mode] += 1

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return counts


def iter_personinfo_never_arrived(xml_path):
    count = 0

    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag != "personinfo":
                elem.clear()
                continue

            duration_v = _safe_float(elem.get("duration"))
            arrival_v = _safe_float(elem.get("arrival"))

            if (duration_v is not None and duration_v < 0) or \
               (arrival_v is not None and arrival_v < 0):
                count += 1

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return count


# ---------------------------------------------------------------------------
# Data collection (now two levels deep: test -> run -> files)
# ---------------------------------------------------------------------------

def collect_raw_records(results_dir):
    """Walk results_dir/<test>/<run>/ and gather raw per-trip records.

    Returns: {test: {run: {epoch: {mode: [metrics_dict, ...]}}}}
    Every metrics_dict has keys matching METRICS.
    """
    tests = {}

    if not os.path.isdir(results_dir):
        return tests

    for test_entry in sorted(os.listdir(results_dir)):
        test_path = os.path.join(results_dir, test_entry)
        if not os.path.isdir(test_path):
            continue

        runs = {}

        for run_entry in sorted(os.listdir(test_path)):
            run_path = os.path.join(test_path, run_entry)
            if not os.path.isdir(run_path):
                continue

            epochs = {}

            for filename in sorted(os.listdir(run_path)):
                trip_match = TRIPINFO_PATTERN.match(filename)
                person_match = PERSONINFO_PATTERN.match(filename)

                if trip_match:
                    epoch = int(trip_match.group(1))
                    xml_path = os.path.join(run_path, filename)
                    for mode, values in iter_tripinfo_records(xml_path):
                        epochs.setdefault(epoch, {m: [] for m in ALL_MODES})
                        epochs[epoch][mode].append(values)

                if person_match:
                    epoch = int(person_match.group(1))
                    xml_path = os.path.join(run_path, filename)
                    for values in iter_personinfo_records(xml_path):
                        epochs.setdefault(epoch, {m: [] for m in ALL_MODES})
                        epochs[epoch][MODE_PED].append(values)

            if epochs:
                runs[run_entry] = epochs

        if runs:
            tests[test_entry] = runs

    return tests


def collect_never_arrived(results_dir):
    """Returns: {test: {run: {epoch: {mode: count}}}}"""
    tests = {}

    if not os.path.isdir(results_dir):
        return tests

    for test_entry in sorted(os.listdir(results_dir)):
        test_path = os.path.join(results_dir, test_entry)
        if not os.path.isdir(test_path):
            continue

        runs = {}

        for run_entry in sorted(os.listdir(test_path)):
            run_path = os.path.join(test_path, run_entry)
            if not os.path.isdir(run_path):
                continue

            epochs = {}
            for filename in sorted(os.listdir(run_path)):
                trip_match = TRIPINFO_PATTERN.match(filename)
                person_match = PERSONINFO_PATTERN.match(filename)
                if not trip_match and not person_match:
                    continue

                if trip_match:
                    epoch = int(trip_match.group(1))
                    xml_path = os.path.join(run_path, filename)
                    entry_counts = epochs.get(epoch, {MODE_CAR: 0, MODE_BIKE: 0, MODE_PED: 0})
                    trip_counts = iter_tripinfo_never_arrived(xml_path)
                    entry_counts[MODE_CAR] += trip_counts.get(MODE_CAR, 0)
                    entry_counts[MODE_BIKE] += trip_counts.get(MODE_BIKE, 0)
                    epochs[epoch] = entry_counts

                if person_match:
                    epoch = int(person_match.group(1))
                    xml_path = os.path.join(run_path, filename)
                    entry_counts = epochs.get(epoch, {MODE_CAR: 0, MODE_BIKE: 0, MODE_PED: 0})
                    entry_counts[MODE_PED] += iter_personinfo_never_arrived(xml_path)
                    epochs[epoch] = entry_counts

            if epochs:
                runs[run_entry] = epochs

        if runs:
            tests[test_entry] = runs

    return tests


# ---------------------------------------------------------------------------
# Derived per-epoch statistics (computed per RUN; same math as the original
# single-run script, just applied once per run instead of once per test)
# ---------------------------------------------------------------------------

def compute_combined_metric_averages(epochs_by_key):
    """Combined (all modes) average per epoch for each metric in METRICS.

    epochs_by_key: {key: {epoch: {mode: [records]}}} where `key` is either a
    test name (original script) or a run name (this script, called per test).
    Returns: {key: {epoch: {metric: avg_value}}}
    """
    result = {}
    for key, epochs in epochs_by_key.items():
        epoch_averages = {}
        for epoch, modes in epochs.items():
            all_values = []
            for mode in ALL_MODES:
                all_values.extend(modes.get(mode, []))

            if not all_values:
                epoch_averages[epoch] = {m: math.nan for m in METRICS}
                continue

            avgs = {}
            for m in METRICS:
                vals = [rec[m] for rec in all_values]
                avgs[m] = sum(vals) / len(vals)
            epoch_averages[epoch] = avgs
        result[key] = epoch_averages
    return result


def compute_combined_throughput(epochs_by_key):
    """Combined throughput (arrivals/hour) per epoch.

    Returns: {key: {epoch: throughput_value}}
    """
    result = {}
    per_hour_factor = EPOCH_DURATION_SECONDS / 3600.0

    for key, epochs in epochs_by_key.items():
        epoch_throughput = {}
        for epoch, modes in epochs.items():
            total_arrivals = sum(len(modes.get(mode, [])) for mode in ALL_MODES)
            epoch_throughput[epoch] = total_arrivals / per_hour_factor if per_hour_factor > 0 else math.nan
        result[key] = epoch_throughput

    return result


def compute_per_mode_wait_stats(epochs_by_key):
    """Per-mode waitingTime statistics per epoch.

    Returns: {key: {mode: {epoch: {
        "total": float, "average": float, "p90": float,
        "variance": float, "throughput": float
    }}}}
    """
    result = {}
    per_hour_factor = EPOCH_DURATION_SECONDS / 3600.0

    for key, epochs in epochs_by_key.items():
        per_mode = {mode: {} for mode in ALL_MODES}

        for epoch, modes in epochs.items():
            for mode in ALL_MODES:
                records = modes.get(mode, [])
                waits = [rec["waitingTime"] for rec in records]

                if not waits:
                    per_mode[mode][epoch] = {
                        "total": math.nan,
                        "average": math.nan,
                        "p90": math.nan,
                        "variance": math.nan,
                        "throughput": 0.0 if per_hour_factor > 0 else math.nan,
                    }
                    continue

                total = sum(waits)
                average = total / len(waits)
                p90 = percentile(waits, 90)
                variance = statistics.pvariance(waits) if len(waits) >= 1 else math.nan
                throughput = len(records) / per_hour_factor if per_hour_factor > 0 else math.nan

                per_mode[mode][epoch] = {
                    "total": total,
                    "average": average,
                    "p90": p90,
                    "variance": variance,
                    "throughput": throughput,
                }

        result[key] = per_mode

    return result


def compute_per_run(raw_records, compute_fn):
    """Apply one of the compute_* functions above (which expect
    {key: {epoch: ...}}) separately to each run within each test.

    raw_records: {test: {run: {epoch: {mode: [...]}}}}
    Returns: {test: {run: <output of compute_fn for that single run>}}
    """
    result = {}
    for test, runs in raw_records.items():
        result[test] = {}
        for run, epochs in runs.items():
            # Wrap the single run's epochs as if it were the only "key" the
            # compute_fn knows about, then unwrap the result.
            single_run_output = compute_fn({run: epochs})[run]
            result[test][run] = single_run_output
    return result


# ---------------------------------------------------------------------------
# Aggregation across runs (mean + error band per test, per epoch)
# ---------------------------------------------------------------------------

def aggregate_combined_averages(per_run_combined_averages):
    """{test: {run: {epoch: {metric: value}}}} -> {test: {metric: {epoch: (mean, err, n)}}}"""
    result = {}
    for test, runs in per_run_combined_averages.items():
        result[test] = {}
        for metric in METRICS:
            per_run_scalar = {
                run: {epoch: vals.get(metric, math.nan) for epoch, vals in epochs.items()}
                for run, epochs in runs.items()
            }
            result[test][metric] = aggregate_scalar_series(per_run_scalar)
    return result


def aggregate_throughput(per_run_throughput):
    """{test: {run: {epoch: value}}} -> {test: {epoch: (mean, err, n)}}"""
    result = {}
    for test, runs in per_run_throughput.items():
        result[test] = aggregate_scalar_series(runs)
    return result


def aggregate_never_arrived(per_run_never_arrived):
    """{test: {run: {epoch: {mode: count}}}} -> {test: {mode: {epoch: (mean, err, n)}}}"""
    result = {}
    for test, runs in per_run_never_arrived.items():
        result[test] = {}
        for mode in ALL_MODES:
            per_run_scalar = {
                run: {epoch: counts.get(mode, 0) for epoch, counts in epochs.items()}
                for run, epochs in runs.items()
            }
            result[test][mode] = aggregate_scalar_series(per_run_scalar)
    return result


def aggregate_per_mode_wait_stats(per_run_wait_stats):
    """per_run_wait_stats: {test: {run: {mode: {epoch: {stat_dict}}}}}

    Returns {test: {mode: {stat_key: {epoch: (mean, err, n)}}}}, where
    stat_key is one of "total", "average", "p90", "variance", "throughput",
    plus "total_smoothed" (each run's total series is rolling-averaged
    first, then aggregated across runs epoch by epoch).
    """
    stat_keys = ("total", "average", "p90", "variance", "throughput")
    result = {}
    for test, runs in per_run_wait_stats.items():
        result[test] = {mode: {} for mode in ALL_MODES}
        for mode in ALL_MODES:
            for stat_key in stat_keys:
                per_run_scalar = {}
                for run, per_mode in runs.items():
                    epochs = per_mode.get(mode, {})
                    per_run_scalar[run] = {e: v[stat_key] for e, v in epochs.items()}
                result[test][mode][stat_key] = aggregate_scalar_series(per_run_scalar)

            # Smoothed total: rolling-average each run's own total series
            # first (so smoothing doesn't blur across independent runs),
            # then aggregate the smoothed values across runs per epoch.
            per_run_smoothed = {}
            for run, per_mode in runs.items():
                epochs = per_mode.get(mode, {})
                xs = sorted(epochs.keys())
                raw = [epochs[x]["total"] for x in xs]
                smoothed = rolling_average(raw, ROLLING_WINDOW)
                per_run_smoothed[run] = dict(zip(xs, smoothed))
            result[test][mode]["total_smoothed"] = aggregate_scalar_series(per_run_smoothed)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _new_figure():
    plt.figure(figsize=(10, 6))


def _finish_plot(xlabel, ylabel, title, output_path):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_series_with_band(epoch_stats, label):
    """epoch_stats: {epoch: (mean, err, n)}. Plots the mean line and a
    shaded +/- err band in the same color."""
    xs = sorted(epoch_stats.keys())
    means = [epoch_stats[x][0] for x in xs]
    errs = [epoch_stats[x][1] for x in xs]

    line, = plt.plot(xs, means, marker="o", linewidth=1.5, label=label)
    color = line.get_color()

    lo, hi = [], []
    for m, e in zip(means, errs):
        if isinstance(m, float) and math.isnan(m):
            lo.append(math.nan)
            hi.append(math.nan)
        else:
            e = 0.0 if (isinstance(e, float) and math.isnan(e)) else e
            lo.append(m - e)
            hi.append(m + e)

    plt.fill_between(xs, lo, hi, alpha=0.2, color=color, linewidth=0)


def plot_combined_metrics(aggregated_combined_averages, plots_dir):
    """duration / waitingTime / timeLoss, combined across modes, mean +/- band across runs."""
    os.makedirs(plots_dir, exist_ok=True)

    for metric in METRICS:
        _new_figure()
        for test, per_metric in aggregated_combined_averages.items():
            _plot_series_with_band(per_metric[metric], test)

        _finish_plot(
            "Epoch", metric,
            f"Average {metric} per epoch (mean +/- {ERROR_BAND_KIND} across runs; cars+bikes+pedestrians combined)",
            os.path.join(plots_dir, f"{metric}.png"),
        )


def plot_combined_throughput(aggregated_throughput, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    _new_figure()
    for test, epoch_stats in aggregated_throughput.items():
        _plot_series_with_band(epoch_stats, test)

    _finish_plot(
        "Epoch", "Throughput (entities/hour)",
        f"Combined throughput per epoch (mean +/- {ERROR_BAND_KIND} across runs; cars+bikes+pedestrians)",
        os.path.join(plots_dir, "throughput_combined.png"),
    )


def plot_never_arrived(aggregated_never_arrived, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    for mode in ALL_MODES:
        _new_figure()
        for test, per_mode in aggregated_never_arrived.items():
            _plot_series_with_band(per_mode[mode], test)

        _finish_plot(
            "Epoch", "never_arrived",
            f"Never arrived (arrival=-1) per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode})",
            os.path.join(plots_dir, f"never_arrived_{mode}.png"),
        )


def plot_queue_metrics(aggregated_queue_metrics, plots_dir):
    """Combined (car+bike) queue length, plus one figure per mode.

    No pedestrian queue column exists in mm_step_metrics.csv, so this only
    covers car and bike.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # --- combined queue (car + bike) ---
    _new_figure()
    for test, per_key in aggregated_queue_metrics.items():
        _plot_series_with_band(per_key["combined"], test)

    _finish_plot(
        "Epoch", "Queue length",
        f"Combined queue length per epoch (car+bike), mean +/- {ERROR_BAND_KIND} across runs",
        os.path.join(plots_dir, "queue_combined.png"),
    )

    # --- per-mode queue length ---
    for mode_key in ("car", "bike"):
        _new_figure()
        for test, per_key in aggregated_queue_metrics.items():
            _plot_series_with_band(per_key[mode_key], test)

        _finish_plot(
            "Epoch", "Queue length",
            f"Queue length per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode_key})",
            os.path.join(plots_dir, f"queue_{mode_key}.png"),
        )


def plot_per_mode_wait_stats(aggregated_wait_stats, plots_dir):
    """One figure per mode per statistic: total (smoothed), average, p90,
    variance, throughput -- each as mean +/- band across runs."""
    os.makedirs(plots_dir, exist_ok=True)

    for mode in ALL_MODES:
        # --- total wait, smoothed with rolling average (per run, then aggregated) ---
        _new_figure()
        for test, per_mode in aggregated_wait_stats.items():
            _plot_series_with_band(per_mode[mode]["total_smoothed"], test)

        _finish_plot(
            "Epoch", "Total waitingTime (rolling avg)",
            f"Total wait per epoch, {ROLLING_WINDOW}-epoch rolling average, mean +/- {ERROR_BAND_KIND} across runs ({mode})",
            os.path.join(plots_dir, f"wait_total_smoothed_{mode}.png"),
        )

        # --- average wait ---
        _new_figure()
        for test, per_mode in aggregated_wait_stats.items():
            _plot_series_with_band(per_mode[mode]["average"], test)

        _finish_plot(
            "Epoch", "Average waitingTime",
            f"Average wait per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode})",
            os.path.join(plots_dir, f"wait_average_{mode}.png"),
        )

        # --- 90th percentile wait ---
        _new_figure()
        for test, per_mode in aggregated_wait_stats.items():
            _plot_series_with_band(per_mode[mode]["p90"], test)

        _finish_plot(
            "Epoch", "90th-percentile waitingTime",
            f"90th-percentile wait per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode})",
            os.path.join(plots_dir, f"wait_p90_{mode}.png"),
        )

        # --- wait stability (variance within epoch, plotted across epochs) ---
        _new_figure()
        for test, per_mode in aggregated_wait_stats.items():
            _plot_series_with_band(per_mode[mode]["variance"], test)

        _finish_plot(
            "Epoch", "Variance of waitingTime within epoch",
            f"Wait stability per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode})",
            os.path.join(plots_dir, f"wait_stability_{mode}.png"),
        )

        # --- throughput per mode ---
        _new_figure()
        for test, per_mode in aggregated_wait_stats.items():
            _plot_series_with_band(per_mode[mode]["throughput"], test)

        _finish_plot(
            "Epoch", "Throughput (entities/hour)",
            f"Throughput per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode})",
            os.path.join(plots_dir, f"throughput_{mode}.png"),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, RESULTS_DIR)
    plots_dir = os.path.join(base_dir, PLOTS_DIR)

    raw_records = collect_raw_records(results_dir)
    never_arrived = collect_never_arrived(results_dir)
    queue_metrics = collect_queue_metrics(results_dir)

    if not raw_records and not never_arrived and not queue_metrics:
        print(
            f"No tripinfo/personinfo/{MM_STEP_METRICS_FILENAME} files found "
            f"under {results_dir}/<test>/<run>/."
        )
        return

    if raw_records:
        per_run_combined_averages = compute_per_run(raw_records, compute_combined_metric_averages)
        per_run_throughput = compute_per_run(raw_records, compute_combined_throughput)
        per_run_wait_stats = compute_per_run(raw_records, compute_per_mode_wait_stats)

        aggregated_combined_averages = aggregate_combined_averages(per_run_combined_averages)
        aggregated_throughput = aggregate_throughput(per_run_throughput)
        aggregated_wait_stats = aggregate_per_mode_wait_stats(per_run_wait_stats)

        plot_combined_metrics(aggregated_combined_averages, plots_dir)
        plot_combined_throughput(aggregated_throughput, plots_dir)
        plot_per_mode_wait_stats(aggregated_wait_stats, plots_dir)

    if never_arrived:
        aggregated_never_arrived = aggregate_never_arrived(never_arrived)
        plot_never_arrived(aggregated_never_arrived, plots_dir)

    if queue_metrics:
        aggregated_queue_metrics = aggregate_queue_metrics(queue_metrics)
        plot_queue_metrics(aggregated_queue_metrics, plots_dir)

    if raw_records:
        run_counts = ", ".join(f"{test}={len(runs)} runs" for test, runs in raw_records.items())
        print(f"Runs found (tripinfo/personinfo): {run_counts}")

    if queue_metrics:
        queue_run_counts = ", ".join(f"{test}={len(runs)} runs" for test, runs in queue_metrics.items())
        print(f"Runs found ({MM_STEP_METRICS_FILENAME}): {queue_run_counts}")

    print(f"Plots saved to {plots_dir}.")


if __name__ == "__main__":
    main()