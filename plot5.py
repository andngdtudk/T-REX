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

# One fixed-name per-episode metrics CSV per run (not per-epoch/decision like
# tripinfo/personinfo). Expected columns include at least: episode,
# car_queue_total, bike_queue_total, ped_queue_total (queues), plus other
# useful training-curve columns such as reward_mean, epsilon, mean_q,
# mean_loss, car/bike/ped_wait_total, bike/ped_wait_frac (see
# EPISODE_METRIC_SPECS below).
# The per-episode metrics CSV may be prefixed with a model/run name, e.g.
# "MPLight_MM_mm_episode_metrics.csv" or plain "mm_episode_metrics.csv".
# Any filename ending in "mm_episode_metrics.csv" is accepted.
EPISODE_METRICS_SUFFIX = "mm_episode_metrics.csv"
EPISODE_METRICS_PATTERN = re.compile(r".*mm_episode_metrics\.csv$")

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
# Episode-level metrics parsing (mm_episode_metrics.csv, one fixed file per
# run, one row per episode). This is the source for queue lengths and for
# a handful of other useful training-curve metrics (reward, epsilon, mean_q,
# mean_loss, wait totals/fractions).
# ---------------------------------------------------------------------------

def read_mm_episode_metrics(csv_path):
    """Read a run's mm_episode_metrics.csv.

    Returns a list of dicts: {"episode": int, <column>: float, ...} for every
    other column in the file (math.nan when missing/unparsable). Rows
    without a usable episode number are skipped. Column set is read
    dynamically, so extra/missing columns across files are tolerated.
    """
    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                episode = _safe_int(raw_row.get("episode"))
                if episode is None:
                    continue

                parsed = {"episode": episode}
                for key, raw_val in raw_row.items():
                    if key == "episode":
                        continue
                    value = _safe_float(raw_val)
                    parsed[key] = value if value is not None else math.nan
                rows.append(parsed)
    except (OSError, csv.Error) as exc:
        print(f"Warning: failed to read {csv_path}: {exc}")

    return rows


def find_run_file(run_path, pattern):
    """Find a file directly inside run_path whose name matches `pattern`
    (a compiled regex tested against the filename, e.g. allowing an
    arbitrary model-name prefix before a fixed suffix).

    Returns the full path to the first match (alphabetically), or None if
    there's no match. Warns if more than one file matches, since only one
    is used.
    """
    try:
        candidates = sorted(
            f for f in os.listdir(run_path)
            if os.path.isfile(os.path.join(run_path, f)) and pattern.match(f)
        )
    except OSError:
        return None

    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"Warning: multiple files matching {pattern.pattern!r} in {run_path}; using {candidates[0]}")
    return os.path.join(run_path, candidates[0])


def collect_episode_metrics(results_dir):
    """Walk results_dir/<test>/<run>/ looking for a file matching
    EPISODE_METRICS_PATTERN (e.g. "mm_episode_metrics.csv" or
    "MPLight_MM_mm_episode_metrics.csv").

    Returns: {test: {run: {episode: {column: value, ...}}}}
    Tests/runs without a matching CSV (or with no usable rows) are simply
    omitted, so this feature degrades gracefully if some runs don't have it.
    If a CSV has duplicate rows for the same episode, the last one wins.
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

            csv_path = find_run_file(run_path, EPISODE_METRICS_PATTERN)
            if csv_path is None:
                continue

            csv_rows = read_mm_episode_metrics(csv_path)
            if not csv_rows:
                continue

            by_episode = {}
            for row in csv_rows:
                by_episode[row["episode"]] = row
            runs[run_entry] = by_episode

        if runs:
            tests[test_entry] = runs

    return tests


def compute_queue_metrics_from_episode(per_run_episode_metrics):
    """Pull car/bike/ped queue totals (+ combined) out of parsed
    mm_episode_metrics.csv rows. Each row is already one total per episode,
    so no within-episode averaging is needed (unlike the old per-decision
    mm_step_metrics.csv source).

    per_run_episode_metrics: {test: {run: {episode: {column: value}}}}
    Returns: {test: {run: {episode: {"car", "bike", "ped", "combined"}}}}
    """
    result = {}
    for test, runs in per_run_episode_metrics.items():
        result[test] = {}
        for run, episodes in runs.items():
            per_episode = {}
            for episode, row in episodes.items():
                car_q = row.get("car_queue_total", math.nan)
                bike_q = row.get("bike_queue_total", math.nan)
                ped_q = row.get("ped_queue_total", math.nan)

                if math.isnan(car_q) and math.isnan(bike_q) and math.isnan(ped_q):
                    combined = math.nan
                else:
                    combined = (0.0 if math.isnan(car_q) else car_q) + \
                               (0.0 if math.isnan(bike_q) else bike_q) + \
                               (0.0 if math.isnan(ped_q) else ped_q)

                per_episode[episode] = {"car": car_q, "bike": bike_q, "ped": ped_q, "combined": combined}
            result[test][run] = per_episode
    return result


def aggregate_queue_metrics(per_run_queue_metrics):
    """{test: {run: {episode: {"car"/"bike"/"ped"/"combined": value}}}}
    -> {test: {"car"/"bike"/"ped"/"combined": {episode: (mean, err, n)}}}
    """
    result = {}
    for test, runs in per_run_queue_metrics.items():
        result[test] = {}
        for key in ("car", "bike", "ped", "combined"):
            per_run_scalar = {
                run: {episode: vals[key] for episode, vals in episodes.items()}
                for run, episodes in runs.items()
            }
            result[test][key] = aggregate_scalar_series(per_run_scalar)
    return result


# Other episode-level metrics worth plotting directly from
# mm_episode_metrics.csv, as (column_name, output_filename_stub, ylabel, title).
# Any spec whose column is absent/all-NaN across every test is skipped
# automatically, so it's safe to list columns that not every run will have.
EPISODE_METRIC_SPECS = (
    ("reward_mean", "reward_mean", "Mean reward", "Mean reward per episode"),
    ("mean_q", "mean_q", "Mean Q-value", "Mean Q-value per episode"),
    ("mean_loss", "mean_loss", "Mean training loss", "Mean training loss per episode"),
    ("epsilon", "epsilon", "Epsilon", "Exploration epsilon per episode"),
    ("bike_wait_frac", "wait_frac_bike", "Bike wait fraction", "Share of total wait time from bikes per episode"),
    ("ped_wait_frac", "wait_frac_ped", "Pedestrian wait fraction", "Share of total wait time from pedestrians per episode"),
    ("car_wait_total", "wait_total_car", "Total wait (car)", "Total car waiting time per episode"),
    ("bike_wait_total", "wait_total_bike", "Total wait (bike)", "Total bike waiting time per episode"),
    ("ped_wait_total", "wait_total_ped", "Total wait (ped)", "Total pedestrian waiting time per episode"),
)


def aggregate_episode_metric_column(per_run_episode_metrics, column):
    """{test: {run: {episode: {column: value, ...}}}} -> {test: {episode: (mean, err, n)}}
    for a single column."""
    result = {}
    for test, runs in per_run_episode_metrics.items():
        per_run_scalar = {
            run: {episode: row.get(column, math.nan) for episode, row in episodes.items()}
            for run, episodes in runs.items()
        }
        result[test] = aggregate_scalar_series(per_run_scalar)
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


def _has_real_data(epoch_stats):
    """True if at least one epoch has a non-NaN mean in this {epoch: (mean, err, n)} dict."""
    return any(
        not (isinstance(mean, float) and math.isnan(mean))
        for mean, _err, _n in epoch_stats.values()
    )


def plot_queue_metrics(aggregated_queue_metrics, plots_dir):
    """Combined (car+bike+ped) queue length, plus one figure per mode.

    A queue key (e.g. "ped") is skipped if no test has any real data for it
    (e.g. the source CSV has no ped_queue_total column at all).
    """
    os.makedirs(plots_dir, exist_ok=True)

    # --- combined queue (car + bike + ped) ---
    if any(_has_real_data(per_key["combined"]) for per_key in aggregated_queue_metrics.values()):
        _new_figure()
        for test, per_key in aggregated_queue_metrics.items():
            _plot_series_with_band(per_key["combined"], test)

        _finish_plot(
            "Epoch", "Queue length",
            f"Combined queue length per epoch (car+bike+ped), mean +/- {ERROR_BAND_KIND} across runs",
            os.path.join(plots_dir, "queue_combined.png"),
        )

    # --- per-mode queue length ---
    for mode_key in ("car", "bike", "ped"):
        if not any(_has_real_data(per_key[mode_key]) for per_key in aggregated_queue_metrics.values()):
            continue

        _new_figure()
        for test, per_key in aggregated_queue_metrics.items():
            _plot_series_with_band(per_key[mode_key], test)

        _finish_plot(
            "Epoch", "Queue length",
            f"Queue length per epoch, mean +/- {ERROR_BAND_KIND} across runs ({mode_key})",
            os.path.join(plots_dir, f"queue_{mode_key}.png"),
        )


def plot_episode_metric_columns(episode_metrics, specs, plots_dir):
    """One figure per spec in EPISODE_METRIC_SPECS, mean +/- band across runs.

    Specs whose column is missing/all-NaN across every test are skipped
    silently (so it's safe to list columns some runs won't have).
    """
    os.makedirs(plots_dir, exist_ok=True)

    for column, filename_stub, ylabel, title in specs:
        aggregated = aggregate_episode_metric_column(episode_metrics, column)

        if not any(_has_real_data(per_epoch) for per_epoch in aggregated.values()):
            continue

        _new_figure()
        for test, per_epoch in aggregated.items():
            _plot_series_with_band(per_epoch, test)

        _finish_plot(
            "Episode", ylabel,
            f"{title}, mean +/- {ERROR_BAND_KIND} across runs",
            os.path.join(plots_dir, f"episode_{filename_stub}.png"),
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
    episode_metrics = collect_episode_metrics(results_dir)

    if not raw_records and not never_arrived and not episode_metrics:
        print(
            f"No tripinfo/personinfo/*{EPISODE_METRICS_SUFFIX} files found "
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

    if episode_metrics:
        queue_metrics = compute_queue_metrics_from_episode(episode_metrics)
        aggregated_queue_metrics = aggregate_queue_metrics(queue_metrics)
        plot_queue_metrics(aggregated_queue_metrics, plots_dir)

        plot_episode_metric_columns(episode_metrics, EPISODE_METRIC_SPECS, plots_dir)

    if raw_records:
        run_counts = ", ".join(f"{test}={len(runs)} runs" for test, runs in raw_records.items())
        print(f"Runs found (tripinfo/personinfo): {run_counts}")

    if episode_metrics:
        episode_run_counts = ", ".join(f"{test}={len(runs)} runs" for test, runs in episode_metrics.items())
        print(f"Runs found (*{EPISODE_METRICS_SUFFIX}): {episode_run_counts}")

    print(f"Plots saved to {plots_dir}.")


if __name__ == "__main__":
    main()