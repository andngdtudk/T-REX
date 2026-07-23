import os
import re
import math
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOTS_DIR = "plots"
TRIPINFO_PATTERN = re.compile(r"^tripinfo_(\d+)\.xml$")
PERSONINFO_PATTERN = re.compile(r"^personinfo_(\d+)\.xml$")
CAR_PREFIX = "car0_"
BIKE_PREFIX = "bike0_"
# Metrics shared by cars, bikes, and pedestrians.
# Note: stopTime is not present on personinfo, so it will be treated as 0
# for pedestrians when computing the combined average.
METRICS = ("duration", "waitingTime", "stopTime", "timeLoss")
PED_METRICS = {"duration", "waitingTime", "timeLoss"}  # available on personinfo
# If True, skip trips that are not finished (arrival == -1 or vaporized == "end").
SKIP_UNFINISHED = True

MODE_CAR = "car"
MODE_BIKE = "bike"
MODE_PED = "pedestrian"


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


# ---------------------------------------------------------------------------
# Tripinfo parsing  (cars + bikes)
# ---------------------------------------------------------------------------

def iter_tripinfo_metrics_by_mode(xml_path):
    """Return per-mode totals and counts from a tripinfo XML file.

    Returns a dict keyed by mode (MODE_CAR / MODE_BIKE), each value being a
    dict with keys 'totals' ({metric: float}) and 'count' (int).
    """
    accum = {
        MODE_CAR:  {"totals": {m: 0.0 for m in METRICS}, "count": 0},
        MODE_BIKE: {"totals": {m: 0.0 for m in METRICS}, "count": 0},
    }

    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag != "tripinfo":
                elem.clear()
                continue

            trip_id = elem.get("id", "")
            vtype   = elem.get("vType", "")

            # Determine mode
            tokens = f"{trip_id} {vtype}".lower()
            if "bike" in tokens or "bicycle" in tokens or "cycle" in tokens:
                mode = MODE_BIKE
            else:
                mode = MODE_CAR

            # Optionally skip unfinished trips
            if SKIP_UNFINISHED:
                arrival   = elem.get("arrival")
                vaporized = elem.get("vaporized")
                if arrival == "-1" or vaporized == "end":
                    elem.clear()
                    continue

            values = {}
            valid  = True
            for metric in METRICS:
                raw   = elem.get(metric)
                value = _safe_float(raw)
                if value is None or value == -1:
                    valid = False
                    break
                values[metric] = value

            if valid:
                bucket = accum[mode]
                for metric, value in values.items():
                    bucket["totals"][metric] += value
                bucket["count"] += 1

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return accum


# ---------------------------------------------------------------------------
# Personinfo parsing  (pedestrians)
# ---------------------------------------------------------------------------

def iter_personinfo_metrics(xml_path):
    """Return totals and count for pedestrian metrics from a personinfo XML.

    stopTime is not present on personinfo elements, so it is counted as 0
    (the key is still included so downstream code stays uniform).
    """
    totals = {m: 0.0 for m in METRICS}
    count  = 0

    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag != "personinfo":
                elem.clear()
                continue

            if SKIP_UNFINISHED:
                arrival  = elem.get("arrival")
                duration = elem.get("duration")
                arrival_v  = _safe_float(arrival)
                duration_v = _safe_float(duration)
                if (arrival_v is not None and arrival_v < 0) or \
                   (duration_v is not None and duration_v < 0):
                    elem.clear()
                    continue

            values = {}
            valid  = True
            for metric in PED_METRICS:
                raw   = elem.get(metric)
                value = _safe_float(raw)
                if value is None or value == -1:
                    valid = False
                    break
                values[metric] = value

            if valid:
                for metric in METRICS:
                    # stopTime is absent for pedestrians – treat as 0
                    totals[metric] += values.get(metric, 0.0)
                count += 1

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return {"totals": totals, "count": count}


# ---------------------------------------------------------------------------
# Legacy single-mode helpers kept for the never-arrived plots
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
            vtype   = elem.get("vType", "")
            mode    = detect_vehicle_mode(trip_id, vtype)

            duration_v = _safe_float(elem.get("duration"))
            arrival_v  = _safe_float(elem.get("arrival"))

            if (duration_v is not None and duration_v < 0) or \
               (arrival_v  is not None and arrival_v  < 0):
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
            arrival_v  = _safe_float(elem.get("arrival"))

            if (duration_v is not None and duration_v < 0) or \
               (arrival_v  is not None and arrival_v  < 0):
                count += 1

            elem.clear()

    except ET.ParseError as exc:
        print(f"Warning: failed to parse {xml_path}: {exc}")

    return count


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_results(results_dir):
    """Collect per-epoch combined averages (car + bike + pedestrian) per method.

    Structure: {method_name: {epoch: {metric: avg_value}}}
    """
    methods = {}

    if not os.path.isdir(results_dir):
        return methods

    for entry in sorted(os.listdir(results_dir)):
        method_path = os.path.join(method_path := os.path.join(results_dir, entry), "")
        method_path = os.path.join(results_dir, entry)
        if not os.path.isdir(method_path):
            continue

        # Gather raw accumulators per epoch across both file types
        # epoch -> {mode: {"totals": {metric: float}, "count": int}}
        epoch_accum = {}

        for filename in sorted(os.listdir(method_path)):
            trip_match   = TRIPINFO_PATTERN.match(filename)
            person_match = PERSONINFO_PATTERN.match(filename)

            if trip_match:
                epoch    = int(trip_match.group(1))
                xml_path = os.path.join(method_path, filename)
                by_mode  = iter_tripinfo_metrics_by_mode(xml_path)
                if epoch not in epoch_accum:
                    epoch_accum[epoch] = {}
                for mode, data in by_mode.items():
                    if mode not in epoch_accum[epoch]:
                        epoch_accum[epoch][mode] = {"totals": {m: 0.0 for m in METRICS}, "count": 0}
                    for m in METRICS:
                        epoch_accum[epoch][mode]["totals"][m] += data["totals"][m]
                    epoch_accum[epoch][mode]["count"] += data["count"]

            if person_match:
                epoch    = int(person_match.group(1))
                xml_path = os.path.join(method_path, filename)
                data     = iter_personinfo_metrics(xml_path)
                if epoch not in epoch_accum:
                    epoch_accum[epoch] = {}
                if MODE_PED not in epoch_accum[epoch]:
                    epoch_accum[epoch][MODE_PED] = {"totals": {m: 0.0 for m in METRICS}, "count": 0}
                for m in METRICS:
                    epoch_accum[epoch][MODE_PED]["totals"][m] += data["totals"][m]
                epoch_accum[epoch][MODE_PED]["count"] += data["count"]

        if not epoch_accum:
            continue

        # Collapse to combined averages
        epoch_averages = {}
        for epoch, modes in epoch_accum.items():
            combined_totals = {m: 0.0 for m in METRICS}
            combined_count  = 0
            for mode_data in modes.values():
                for m in METRICS:
                    combined_totals[m] += mode_data["totals"][m]
                combined_count += mode_data["count"]

            if combined_count == 0:
                epoch_averages[epoch] = {m: math.nan for m in METRICS}
            else:
                epoch_averages[epoch] = {m: combined_totals[m] / combined_count for m in METRICS}

        methods[entry] = epoch_averages

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
            trip_match   = TRIPINFO_PATTERN.match(filename)
            person_match = PERSONINFO_PATTERN.match(filename)
            if not trip_match and not person_match:
                continue

            if trip_match:
                epoch    = int(trip_match.group(1))
                xml_path = os.path.join(method_path, filename)
                entry_counts = epochs.get(epoch, {MODE_CAR: 0, MODE_BIKE: 0, MODE_PED: 0})
                trip_counts  = iter_tripinfo_never_arrived(xml_path)
                entry_counts[MODE_CAR]  += trip_counts.get(MODE_CAR,  0)
                entry_counts[MODE_BIKE] += trip_counts.get(MODE_BIKE, 0)
                epochs[epoch] = entry_counts

            if person_match:
                epoch    = int(person_match.group(1))
                xml_path = os.path.join(method_path, filename)
                entry_counts = epochs.get(epoch, {MODE_CAR: 0, MODE_BIKE: 0, MODE_PED: 0})
                entry_counts[MODE_PED] += iter_personinfo_never_arrived(xml_path)
                epochs[epoch] = entry_counts

        if epochs:
            methods[entry] = epochs

    return methods


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(methods, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        for method, epochs in methods.items():
            xs = sorted(epochs.keys())
            ys = [epochs[x].get(metric, math.nan) for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Average {metric} per epoch (cars + bikes + pedestrians combined)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = os.path.join(plots_dir, f"{metric}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()


def plot_never_arrived(methods, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    for mode in (MODE_CAR, MODE_BIKE, MODE_PED):
        plt.figure(figsize=(10, 6))
        for method, epochs in methods.items():
            xs = sorted(epochs.keys())
            ys = [epochs[x].get(mode, 0) for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)

        plt.xlabel("Epoch")
        plt.ylabel("never_arrived")
        plt.title(f"Never arrived (arrival=-1) per epoch ({mode})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = os.path.join(plots_dir, f"never_arrived_{mode}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, RESULTS_DIR)
    plots_dir   = os.path.join(base_dir, PLOTS_DIR)

    methods       = collect_results(results_dir)
    never_arrived = collect_never_arrived(results_dir)
    if not methods and not never_arrived:
        print(f"No tripinfo/personinfo files found in {results_dir}.")
        return

    if methods:
        plot_metrics(methods, plots_dir)
    if never_arrived:
        plot_never_arrived(never_arrived, plots_dir)

    print(f"Plots saved to {plots_dir}.")


if __name__ == "__main__":
    main()