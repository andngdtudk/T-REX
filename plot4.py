import os
import re
import math
import statistics
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOTS_DIR = "plots"
TRIPINFO_PATTERN = re.compile(r"^tripinfo_(\d+)\.xml$")
PERSONINFO_PATTERN = re.compile(r"^personinfo_(\d+)\.xml$")

# Combined metrics plotted directly from tripinfo/personinfo attributes.
# stopTime is intentionally excluded (not plotted per latest instructions).
METRICS = ("duration", "waitingTime", "timeLoss")
PED_METRICS = {"duration", "waitingTime", "timeLoss"}  # fields available on personinfo

# If True, skip trips that are not finished (arrival == -1 or vaporized == "end").
SKIP_UNFINISHED = True

# Window (in seconds) used as the denominator for throughput calculations.
# throughput = (# arrivals in epoch) / (EPOCH_DURATION_SECONDS / 3600)
EPOCH_DURATION_SECONDS = 3600

# Window size (in epochs) for rolling-average smoothing plots.
ROLLING_WINDOW = 5

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
# Data collection
# ---------------------------------------------------------------------------

def collect_raw_records(results_dir):
    """Walk results_dir and gather raw per-trip records.

    Returns: {method: {epoch: {mode: [metrics_dict, ...]}}}
    Every metrics_dict has keys matching METRICS.
    """
    methods = {}

    if not os.path.isdir(results_dir):
        return methods

    for entry in sorted(os.listdir(results_dir)):
        method_path = os.path.join(results_dir, entry)
        if not os.path.isdir(method_path):
            continue

        epochs = {}

        for filename in sorted(os.listdir(method_path)):
            trip_match = TRIPINFO_PATTERN.match(filename)
            person_match = PERSONINFO_PATTERN.match(filename)

            if trip_match:
                epoch = int(trip_match.group(1))
                xml_path = os.path.join(method_path, filename)
                for mode, values in iter_tripinfo_records(xml_path):
                    epochs.setdefault(epoch, {m: [] for m in ALL_MODES})
                    epochs[epoch][mode].append(values)

            if person_match:
                epoch = int(person_match.group(1))
                xml_path = os.path.join(method_path, filename)
                for values in iter_personinfo_records(xml_path):
                    epochs.setdefault(epoch, {m: [] for m in ALL_MODES})
                    epochs[epoch][MODE_PED].append(values)

        if epochs:
            methods[entry] = epochs

    return methods


def collect_never_arrived(results_dir):
    methods = {}

    if not os.path.isdir(results_dir):
        return methods

    for entry in sorted(os.listdir(results_dir)):
        method_path = os.path.join(results_dir, entry)
        if not os.path.isdir(method_path):
            continue

        epochs = {}
        for filename in sorted(os.listdir(method_path)):
            trip_match = TRIPINFO_PATTERN.match(filename)
            person_match = PERSONINFO_PATTERN.match(filename)
            if not trip_match and not person_match:
                continue

            if trip_match:
                epoch = int(trip_match.group(1))
                xml_path = os.path.join(method_path, filename)
                entry_counts = epochs.get(epoch, {MODE_CAR: 0, MODE_BIKE: 0, MODE_PED: 0})
                trip_counts = iter_tripinfo_never_arrived(xml_path)
                entry_counts[MODE_CAR] += trip_counts.get(MODE_CAR, 0)
                entry_counts[MODE_BIKE] += trip_counts.get(MODE_BIKE, 0)
                epochs[epoch] = entry_counts

            if person_match:
                epoch = int(person_match.group(1))
                xml_path = os.path.join(method_path, filename)
                entry_counts = epochs.get(epoch, {MODE_CAR: 0, MODE_BIKE: 0, MODE_PED: 0})
                entry_counts[MODE_PED] += iter_personinfo_never_arrived(xml_path)
                epochs[epoch] = entry_counts

        if epochs:
            methods[entry] = epochs

    return methods


# ---------------------------------------------------------------------------
# Derived per-epoch statistics
# ---------------------------------------------------------------------------

def compute_combined_metric_averages(raw_records):
    """Combined (all modes) average per epoch for each metric in METRICS.

    Returns: {method: {epoch: {metric: avg_value}}}
    """
    result = {}
    for method, epochs in raw_records.items():
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
        result[method] = epoch_averages
    return result


def compute_combined_throughput(raw_records):
    """Combined throughput (arrivals/hour) per epoch.

    Returns: {method: {epoch: throughput_value}}
    """
    result = {}
    per_hour_factor = EPOCH_DURATION_SECONDS / 3600.0

    for method, epochs in raw_records.items():
        epoch_throughput = {}
        for epoch, modes in epochs.items():
            total_arrivals = sum(len(modes.get(mode, [])) for mode in ALL_MODES)
            epoch_throughput[epoch] = total_arrivals / per_hour_factor if per_hour_factor > 0 else math.nan
        result[method] = epoch_throughput

    return result


def compute_per_mode_wait_stats(raw_records):
    """Per-mode waitingTime statistics per epoch.

    Returns: {method: {mode: {epoch: {
        "total": float, "average": float, "p90": float,
        "variance": float, "throughput": float
    }}}}
    """
    result = {}
    per_hour_factor = EPOCH_DURATION_SECONDS / 3600.0

    for method, epochs in raw_records.items():
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

        result[method] = per_mode

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


def plot_combined_metrics(combined_averages, plots_dir):
    """duration / waitingTime / timeLoss, combined across modes."""
    os.makedirs(plots_dir, exist_ok=True)

    for metric in METRICS:
        _new_figure()
        for method, epochs in combined_averages.items():
            xs = sorted(epochs.keys())
            ys = [epochs[x].get(metric, math.nan) for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", metric,
            f"Average {metric} per epoch (cars + bikes + pedestrians combined)",
            os.path.join(plots_dir, f"{metric}.png"),
        )


def plot_combined_throughput(combined_throughput, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    _new_figure()
    for method, epochs in combined_throughput.items():
        xs = sorted(epochs.keys())
        ys = [epochs[x] for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

    _finish_plot(
        "Epoch", "Throughput (entities/hour)",
        "Combined throughput per epoch (cars + bikes + pedestrians)",
        os.path.join(plots_dir, "throughput_combined.png"),
    )


def plot_never_arrived(methods, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    for mode in ALL_MODES:
        _new_figure()
        for method, epochs in methods.items():
            xs = sorted(epochs.keys())
            ys = [epochs[x].get(mode, 0) for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", "never_arrived",
            f"Never arrived (arrival=-1) per epoch ({mode})",
            os.path.join(plots_dir, f"never_arrived_{mode}.png"),
        )


def plot_per_mode_wait_stats(per_mode_stats, plots_dir):
    """One figure per mode per statistic: total (smoothed), average, p90,
    variance, throughput.
    """
    os.makedirs(plots_dir, exist_ok=True)

    for mode in ALL_MODES:
        # --- total wait, smoothed with rolling average ---
        _new_figure()
        for method, per_mode in per_mode_stats.items():
            epochs = per_mode.get(mode, {})
            xs = sorted(epochs.keys())
            ys_raw = [epochs[x]["total"] for x in xs]
            ys_smoothed = rolling_average(ys_raw, ROLLING_WINDOW)
            plt.plot(xs, ys_smoothed, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", "Total waitingTime (rolling avg)",
            f"Total wait per epoch, {ROLLING_WINDOW}-epoch rolling average ({mode})",
            os.path.join(plots_dir, f"wait_total_smoothed_{mode}.png"),
        )

        # --- average wait ---
        _new_figure()
        for method, per_mode in per_mode_stats.items():
            epochs = per_mode.get(mode, {})
            xs = sorted(epochs.keys())
            ys = [epochs[x]["average"] for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", "Average waitingTime",
            f"Average wait per epoch ({mode})",
            os.path.join(plots_dir, f"wait_average_{mode}.png"),
        )

        # --- 90th percentile wait ---
        _new_figure()
        for method, per_mode in per_mode_stats.items():
            epochs = per_mode.get(mode, {})
            xs = sorted(epochs.keys())
            ys = [epochs[x]["p90"] for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", "90th-percentile waitingTime",
            f"90th-percentile wait per epoch ({mode})",
            os.path.join(plots_dir, f"wait_p90_{mode}.png"),
        )

        # --- wait stability (variance within epoch, plotted across epochs) ---
        _new_figure()
        for method, per_mode in per_mode_stats.items():
            epochs = per_mode.get(mode, {})
            xs = sorted(epochs.keys())
            ys = [epochs[x]["variance"] for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", "Variance of waitingTime within epoch",
            f"Wait stability per epoch ({mode})",
            os.path.join(plots_dir, f"wait_stability_{mode}.png"),
        )

        # --- throughput per mode ---
        _new_figure()
        for method, per_mode in per_mode_stats.items():
            epochs = per_mode.get(mode, {})
            xs = sorted(epochs.keys())
            ys = [epochs[x]["throughput"] for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        _finish_plot(
            "Epoch", "Throughput (entities/hour)",
            f"Throughput per epoch ({mode})",
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

    if not raw_records and not never_arrived:
        print(f"No tripinfo/personinfo files found in {results_dir}.")
        return

    if raw_records:
        combined_averages = compute_combined_metric_averages(raw_records)
        combined_throughput = compute_combined_throughput(raw_records)
        per_mode_stats = compute_per_mode_wait_stats(raw_records)

        plot_combined_metrics(combined_averages, plots_dir)
        plot_combined_throughput(combined_throughput, plots_dir)
        plot_per_mode_wait_stats(per_mode_stats, plots_dir)

    if never_arrived:
        plot_never_arrived(never_arrived, plots_dir)

    print(f"Plots saved to {plots_dir}.")


if __name__ == "__main__":
    main()