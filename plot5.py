import os
import re
import csv
import math
import argparse
import statistics
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
#import scienceplots

#plt.style.use('science')


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
# Special-case test/run categories for plotting
# ---------------------------------------------------------------------------
# Baselines (FIXEDTIME/MAXPRESSURE/MAXWAVE/STOCHASTIC) don't have a
# meaningful "training curve" over epochs, so instead of a noisy line they're
# shown as a flat horizontal dashed line at their average value across
# epochs.
HLINE_PREFIXES = ("FIXEDTIME", "MAXPRESSURE", "MAXWAVE", "STOCHASTIC")

# Long-horizon RL methods (IPPO/FMA2C) are trained across 1400 epochs, which
# would otherwise squash/dwarf the ~100-epoch runs sharing the same axes. By
# default they're collapsed into a single marker (the average of their last
# few epochs) placed to the right of the other tests' epoch range. Pass
# --longones on the command line to plot them normally instead.
MARKER_PREFIXES = ("IPPO", "FMA2C")

# Set from argv in main(); read by classify_test().
LONGONES = False

# One shared colour palette, assigned per-test across an entire figure (not
# per-category), so a "normal" line, a baseline hline, and a long-horizon
# marker on the same plot never end up sharing a colour. Cycled through in
# whatever order the tests are collected (alphabetical, since they come from
# sorted(os.listdir(...))).
PLOT_COLOR_PALETTE = [
    *plt.get_cmap("tab20").colors,
    *plt.get_cmap("tab20b").colors,
]

# Linestyles / marker shapes cycled (by index, within their own category
# only) so multiple hlines -- or multiple long-horizon markers -- sharing a
# plot are still visually distinguishable from each other, on top of having
# distinct colours.
HLINE_LINESTYLES = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
MARKER_STYLES = ["^", "s", "D", "v", "P", "X", "o", "*"]


def classify_test(test_name):
    """Classify a test/run name into "hline", "marker", or "normal" for
    plotting purposes, based on its prefix (case-insensitive)."""
    upper = test_name.upper()
    if upper.startswith(HLINE_PREFIXES):
        return "hline"
    if upper.startswith(MARKER_PREFIXES) and not LONGONES:
        return "marker"
    return "normal"


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
    stat_key is one of "total", "average", "p90", "variance", "throughput".
    (Rolling-average smoothing is applied uniformly at plot time instead of
    here, so every plot -- not just this one -- gets the same treatment.)
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


# --- Broken/split y-axis support -------------------------------------------
# For plots where one or two tests have rare, very large excursions (e.g. a
# terrible first few epochs) that would otherwise flatten everything else
# onto a single axis, we can render a small top panel showing the full range
# and a larger bottom panel zoomed into where most of the data lives.

def _new_split_figure(height_ratios=(1, 3)):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )
    return fig, ax_top, ax_bot


def _add_break_marks(ax_top, ax_bot):
    """Draw the small diagonal '//' marks on the shared border between the
    two stacked axes, the standard matplotlib broken-axis convention."""
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False, labelbottom=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.012  # diagonal mark size, in axes-fraction coordinates
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def _auto_split_ylims(per_test_epoch_stats, break_value, top_pad=1.05):
    """Compute (top_ylim, bottom_ylim) for a split-axis plot: the bottom,
    zoomed-in panel covers [0, break_value]; the top panel covers
    [break_value, actual max] so nothing is clipped. Uses mean+err (not just
    the mean) so shaded bands aren't cut off either."""
    all_his = []
    for epoch_stats in per_test_epoch_stats.values():
        for m, e, _n in epoch_stats.values():
            if isinstance(m, float) and math.isnan(m):
                continue
            e = 0.0 if (isinstance(e, float) and math.isnan(e)) else e
            all_his.append(m + e)

    data_max = max(all_his) if all_his else break_value * 2
    bottom_ylim = (0, break_value)
    top_ylim = (break_value, max(data_max * top_pad, break_value * 1.1))
    return top_ylim, bottom_ylim


def _finish_split_plot(ax_top, ax_bot, xlabel, ylabel, title, output_path,
                        top_ylim, bottom_ylim):
    ax_top.set_ylim(*top_ylim)
    ax_bot.set_ylim(*bottom_ylim)

    _add_break_marks(ax_top, ax_bot)

    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel(ylabel)
    ax_top.set_title(title)

    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax_bot.get_legend_handles_labels()
    ax_top.legend(handles, labels, loc="upper right", fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _new_split_figure_and_plot(per_test_series, break_value, xlabel, ylabel,
                                title, output_path, smooth=True):
    """Convenience wrapper: build a split-axis figure, plot the same series
    on both panels, auto-size the panels around `break_value`, and save.
    Drop-in alternative to _new_figure() + _plot_multi_test_series() +
    _finish_plot() for metrics that need a broken axis."""
    _fig, ax_top, ax_bot = _new_split_figure()
    _plot_multi_test_series(per_test_series, smooth=smooth, axes=(ax_top, ax_bot))

    top_ylim, bottom_ylim = _auto_split_ylims(per_test_series, break_value)
    _finish_split_plot(ax_top, ax_bot, xlabel, ylabel, title, output_path, top_ylim, bottom_ylim)


def _smooth_epoch_stats(epoch_stats, window=ROLLING_WINDOW):
    """epoch_stats: {epoch: (mean, err, n)}. Returns a new dict with the
    mean and err series each replaced by their rolling average over
    `window` epochs (n is left untouched)."""
    xs = sorted(epoch_stats.keys())
    means = [epoch_stats[x][0] for x in xs]
    errs = [epoch_stats[x][1] for x in xs]
    ns = [epoch_stats[x][2] for x in xs]

    smoothed_means = rolling_average(means, window)
    smoothed_errs = rolling_average(errs, window)

    return {x: (m, e, n) for x, m, e, n in zip(xs, smoothed_means, smoothed_errs, ns)}


def _plot_series_with_band(epoch_stats, label, smooth=True, color=None, ax=None):
    """epoch_stats: {epoch: (mean, err, n)}. Plots the mean line and a
    shaded +/- err band in the same color. If `smooth` is True (default),
    both the mean and the error band are rolling-averaged over
    ROLLING_WINDOW epochs first. `ax` defaults to the current axes."""
    ax = ax if ax is not None else plt.gca()

    if smooth:
        epoch_stats = _smooth_epoch_stats(epoch_stats, ROLLING_WINDOW)

    xs = sorted(epoch_stats.keys())
    means = [epoch_stats[x][0] for x in xs]
    errs = [epoch_stats[x][1] for x in xs]

    line, = ax.plot(xs, means, marker="o", linewidth=1.5, label=label, color=color)
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

    ax.fill_between(xs, lo, hi, alpha=0.2, color=color, linewidth=0)


def _plot_hline_series(epoch_stats, label, style_index=0, color=None, ax=None):
    """epoch_stats: {epoch: (mean, err, n)}. Plots a horizontal line at the
    average of the (unsmoothed) per-epoch means -- for baseline tests that
    don't have a meaningful epoch-by-epoch trend. `style_index` cycles
    through HLINE_LINESTYLES so multiple baselines on the same plot are
    distinguishable even where their colours are close. `ax` defaults to
    the current axes."""
    ax = ax if ax is not None else plt.gca()

    means = [
        m for m, _e, _n in epoch_stats.values()
        if not (isinstance(m, float) and math.isnan(m))
    ]
    if not means:
        return
    avg = sum(means) / len(means)
    linestyle = HLINE_LINESTYLES[style_index % len(HLINE_LINESTYLES)]
    ax.axhline(y=avg, linestyle=linestyle, linewidth=1.5, label=label, color=color)


def _plot_marker_series(epoch_stats, label, x_pos, window=ROLLING_WINDOW,
                         style_index=0, color=None, ax=None):
    """epoch_stats: {epoch: (mean, err, n)}. Plots a single marker at x_pos
    whose y value is the average of the last `window` epochs' means -- for
    long-horizon tests that are collapsed to a single point. `style_index`
    cycles through MARKER_STYLES so multiple long-horizon tests on the same
    plot are distinguishable. `ax` defaults to the current axes."""
    ax = ax if ax is not None else plt.gca()

    xs = sorted(epoch_stats.keys())
    if not xs:
        return
    last_xs = xs[-window:] if len(xs) >= window else xs
    vals = [
        epoch_stats[x][0] for x in last_xs
        if not (isinstance(epoch_stats[x][0], float) and math.isnan(epoch_stats[x][0]))
    ]
    if not vals:
        return
    avg = sum(vals) / len(vals)
    marker = MARKER_STYLES[style_index % len(MARKER_STYLES)]
    ax.plot(x_pos, avg, marker=marker, markersize=11, linestyle="none", label=label, color=color)


def _plot_multi_test_series(per_test_epoch_stats, smooth=True, ax=None, axes=None):
    """per_test_epoch_stats: {test: {epoch: (mean, err, n)}}.

    Dispatches each test to the appropriate plotting style based on
    classify_test(): a normal smoothed line+band, a horizontal average line
    (baselines), or a single marker placed to the right of the other
    (normal) tests' epoch range (long-horizon RL methods). Every test gets a
    unique colour from PLOT_COLOR_PALETTE (shared across categories, so a
    normal line, an hline, and a marker never collide), plus a cycled
    linestyle/marker-shape within its own category for extra distinction.

    `ax`: a single axes to draw on (normal, single-panel plots).
    `axes`: an iterable of axes to draw the *same* series on all of them
    (used for broken/split-axis plots, e.g. (ax_top, ax_bot)).
    If neither is given, uses plt.gca().
    """
    target_axes = list(axes) if axes is not None else [ax if ax is not None else plt.gca()]

    categories = {test: classify_test(test) for test in per_test_epoch_stats}

    normal_epochs = set()
    for test, cat in categories.items():
        if cat == "normal":
            normal_epochs.update(per_test_epoch_stats[test].keys())

    if normal_epochs:
        x_max = max(normal_epochs)
    else:
        # No "normal" tests on this plot; fall back to the max epoch across
        # everything so the marker still has a sensible x position.
        all_epochs = set()
        for epochs in per_test_epoch_stats.values():
            all_epochs.update(epochs.keys())
        x_max = max(all_epochs) if all_epochs else 100

    marker_x = x_max + max(10, 0.1 * x_max)

    color_i = 0
    hline_i = 0
    marker_i = 0
    for test, epoch_stats in per_test_epoch_stats.items():
        cat = categories[test]
        color = PLOT_COLOR_PALETTE[color_i % len(PLOT_COLOR_PALETTE)]
        color_i += 1

        for target_ax in target_axes:
            if cat == "hline":
                _plot_hline_series(epoch_stats, test, style_index=hline_i, color=color, ax=target_ax)
            elif cat == "marker":
                _plot_marker_series(epoch_stats, test, marker_x, style_index=marker_i, color=color, ax=target_ax)
            else:
                _plot_series_with_band(epoch_stats, test, smooth=smooth, color=color, ax=target_ax)

        if cat == "hline":
            hline_i += 1
        elif cat == "marker":
            marker_i += 1


# Metrics whose combined-metric plot should use a broken/split y-axis (a
# small top panel for the full range, a larger zoomed-in bottom panel)
# because one or two tests have rare, very large early-training excursions
# that would otherwise flatten everything else onto a single axis. Maps
# metric -> break value (data at/below this goes in the zoomed bottom
# panel; above it goes in the top panel). A metric not listed here is
# plotted on a single normal axis. Tune the break value to just above where
# the "interesting" cluster of methods lives.
SPLIT_AXIS_BREAKPOINTS = {
    "duration": 200,
    "timeLoss": 200,
    "waitingTime": 200,
}


def plot_combined_metrics(aggregated_combined_averages, plots_dir):
    """duration / waitingTime / timeLoss, combined across modes, mean +/- band across runs."""
    os.makedirs(plots_dir, exist_ok=True)

    for metric in METRICS:
        per_test_series = {
            test: per_metric[metric]
            for test, per_metric in aggregated_combined_averages.items()
        }

        if metric == "duration":
            metric_axis = "Duration (s)"
            metric_label = "duration"
        elif metric == "waitingTime":
            metric_axis = "Waiting time (s)"
            metric_label = "waiting time"
        elif metric == "timeLoss":
            metric_axis = "Time loss (s)"
            metric_label     = "time loss"
        elif metric == "never_arrived":
            metric_axis = "Never arrived (count)"
            metric_label = "never arrived entities"
        else:
            metric_axis = metric
            metric_label = metric

        title = f"Average combined {metric_label} per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs"
        output_path = os.path.join(plots_dir, f"{metric}.png")

        break_value = SPLIT_AXIS_BREAKPOINTS.get(metric)
        if break_value is not None:
            _new_split_figure_and_plot(
                per_test_series, break_value, "Epoch", metric_axis, title, output_path,
            )
        else:
            _new_figure()
            _plot_multi_test_series(per_test_series)
            _finish_plot("Epoch", metric_axis, title, output_path)


def plot_combined_throughput(aggregated_throughput, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    _new_figure()
    _plot_multi_test_series(aggregated_throughput)

    _finish_plot(
        "Epoch", "Throughput (entities/hour)",
        f"Combined throughput per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs",
        os.path.join(plots_dir, "throughput_combined.png"),
    )


def plot_never_arrived(aggregated_never_arrived, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    for mode in ALL_MODES:
        _new_figure()
        per_test_series = {
            test: per_mode[mode] for test, per_mode in aggregated_never_arrived.items()
        }
        _plot_multi_test_series(per_test_series)

        if mode == MODE_PED or mode == "ped":
            mode_label = "pedestrians"
        elif mode == MODE_BIKE or mode == "bike":
            mode_label = "cyclists"
        else:
            mode_label = "cars"

        _finish_plot(
            "Epoch", "Never arrived (entities)",
            f"Never arrived per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
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
        per_test_series = {
            test: per_key["combined"] for test, per_key in aggregated_queue_metrics.items()
        }
        _plot_multi_test_series(per_test_series)

        _finish_plot(
            "Epoch", "Queue length",
            f"Combined queue$^*$ length per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs",
            os.path.join(plots_dir, "queue_combined.png"),
        )

    # --- per-mode queue length ---
    for mode_key in ("car", "bike", "ped"):
        if not any(_has_real_data(per_key[mode_key]) for per_key in aggregated_queue_metrics.values()):
            continue

        _new_figure()
        per_test_series = {
            test: per_key[mode_key] for test, per_key in aggregated_queue_metrics.items()
        }
        _plot_multi_test_series(per_test_series)

        if mode_key == "ped":
            mode_label = "pedestrians"
            _finish_plot(
            "Epoch", "Queue length",
            f"Queue$^*$ length per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
            os.path.join(plots_dir, f"queue_{mode_key}.png"),
        )
        else:
            if mode_key == "bike":
                mode_label = "cyclists"
            else:
                mode_label = "cars"
            _finish_plot(
                "Epoch", "Queue length",
                f"Queue length per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
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
        _plot_multi_test_series(aggregated)

        _finish_plot(
            "Episode", ylabel,
            f"{title}, mean $+/-$ {ERROR_BAND_KIND} across runs",
            os.path.join(plots_dir, f"episode_{filename_stub}.png"),
        )


def plot_per_mode_wait_stats(aggregated_wait_stats, plots_dir):
    """One figure per mode per statistic: total, average, p90, variance,
    throughput -- each as mean +/- band across runs (rolling-averaged)."""
    os.makedirs(plots_dir, exist_ok=True)

    for mode in ALL_MODES:
        if mode == MODE_PED or mode == "ped":
            mode_label = "pedestrians"
        elif mode == MODE_BIKE or mode == "bike":
            mode_label = "cyclists"
        else:
            mode_label = "cars"

        # --- total wait ---
        _new_figure()
        per_test_series = {
            test: per_mode[mode]["total"] for test, per_mode in aggregated_wait_stats.items()
        }
        _plot_multi_test_series(per_test_series)

        _finish_plot(
            "Epoch", "Total waiting time (s)",
            f"Total wait per epoch, {ROLLING_WINDOW}-epoch rolling average, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
            os.path.join(plots_dir, f"wait_total_smoothed_{mode}.png"),
        )

        # --- average wait ---
        _new_figure()
        per_test_series = {
            test: per_mode[mode]["average"] for test, per_mode in aggregated_wait_stats.items()
        }
        _plot_multi_test_series(per_test_series)

        _finish_plot(
            "Epoch", "Average waiting time (s)",
            f"Average wait per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
            os.path.join(plots_dir, f"wait_average_{mode}.png"),
        )

        # --- 90th percentile wait ---
        _new_figure()
        per_test_series = {
            test: per_mode[mode]["p90"] for test, per_mode in aggregated_wait_stats.items()
        }
        _plot_multi_test_series(per_test_series)

        _finish_plot(
            "Epoch", "90th-percentile waiting time (s)",
            f"90th-percentile wait per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
            os.path.join(plots_dir, f"wait_p90_{mode}.png"),
        )

        # --- wait stability (variance within epoch, plotted across epochs) ---
        _new_figure()
        per_test_series = {
            test: per_mode[mode]["variance"] for test, per_mode in aggregated_wait_stats.items()
        }
        _plot_multi_test_series(per_test_series)

        _finish_plot(
            "Epoch", "Variance of waiting time within epoch",
            f"Wait stability per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
            os.path.join(plots_dir, f"wait_stability_{mode}.png"),
        )

        # --- throughput per mode ---
        _new_figure()
        per_test_series = {
            test: per_mode[mode]["throughput"] for test, per_mode in aggregated_wait_stats.items()
        }
        _plot_multi_test_series(per_test_series)

        _finish_plot(
            "Epoch", "Throughput (entities/hour)",
            f"Throughput per epoch, mean $+/-$ {ERROR_BAND_KIND} across runs ({mode_label})",
            os.path.join(plots_dir, f"throughput_{mode}.png"),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot MoveMax results.")
    parser.add_argument(
        "--longones",
        action="store_true",
        help=(
            "Plot IPPO/FMA2C runs normally (full 1400-epoch line+band) "
            "instead of collapsing them into a single marker to the right "
            "of the other tests."
        ),
    )
    return parser.parse_args()


def main():
    global LONGONES
    args = parse_args()
    LONGONES = args.longones

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