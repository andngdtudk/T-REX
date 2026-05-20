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
METRICS = ("duration", "waitingTime", "stopTime", "timeLoss")
# If True, skip trips that are not finished (arrival == -1 or vaporized == "end").
SKIP_UNFINISHED = True

MODE_CAR = "car"
MODE_BIKE = "bike"
MODE_PED = "pedestrian"


def iter_tripinfo_metrics(xml_path):
	totals = {m: 0.0 for m in METRICS}
	count = 0

	for _event, elem in ET.iterparse(xml_path, events=("end",)):
		if elem.tag != "tripinfo":
			continue

		trip_id = elem.get("id", "")
		if not trip_id.startswith(CAR_PREFIX):
			elem.clear()
			continue

		arrival = elem.get("arrival")
		vaporized = elem.get("vaporized")
		if SKIP_UNFINISHED and (arrival == "-1" or vaporized == "end"):
			elem.clear()
			continue

		values = {}
		valid = True
		for metric in METRICS:
			raw = elem.get(metric)
			if raw is None:
				valid = False
				break
			try:
				value = float(raw)
			except ValueError:
				valid = False
				break
			if value == -1:
				valid = False
				break
			values[metric] = value

		if valid:
			for metric, value in values.items():
				totals[metric] += value
			count += 1

		elem.clear()

	if count == 0:
		return {m: math.nan for m in METRICS}

	return {m: totals[m] / count for m in METRICS}


def detect_vehicle_mode(trip_id, vtype):
	tokens = f"{trip_id or ''} {vtype or ''}".lower()
	if "bike" in tokens or "bicycle" in tokens or "cycle" in tokens:
		return MODE_BIKE
	return MODE_CAR


def iter_tripinfo_never_arrived(xml_path):
	counts = {MODE_CAR: 0, MODE_BIKE: 0}

	for _event, elem in ET.iterparse(xml_path, events=("end",)):
		if elem.tag != "tripinfo":
			continue

		trip_id = elem.get("id", "")
		vtype = elem.get("vType", "")
		mode = detect_vehicle_mode(trip_id, vtype)

		duration = elem.get("duration")
		arrival = elem.get("arrival")
		try:
			duration_value = float(duration) if duration is not None else None
		except ValueError:
			duration_value = None
		try:
			arrival_value = float(arrival) if arrival is not None else None
		except ValueError:
			arrival_value = None

		if duration_value is not None and duration_value < 0:
			counts[mode] += 1
		elif arrival_value is not None and arrival_value < 0:
			counts[mode] += 1

		elem.clear()

	return counts


def iter_personinfo_never_arrived(xml_path):
	count = 0

	for _event, elem in ET.iterparse(xml_path, events=("end",)):
		if elem.tag != "personinfo":
			continue

		duration = elem.get("duration")
		arrival = elem.get("arrival")
		try:
			duration_value = float(duration) if duration is not None else None
		except ValueError:
			duration_value = None
		try:
			arrival_value = float(arrival) if arrival is not None else None
		except ValueError:
			arrival_value = None

		if duration_value is not None and duration_value < 0:
			count += 1
		elif arrival_value is not None and arrival_value < 0:
			count += 1

		elem.clear()

	return count


def collect_results(results_dir):
	methods = {}

	if not os.path.isdir(results_dir):
		return methods

	for entry in sorted(os.listdir(results_dir)):
		method_path = os.path.join(results_dir, entry)
		if not os.path.isdir(method_path):
			continue

		epochs = {}
		for filename in sorted(os.listdir(method_path)):
			match = TRIPINFO_PATTERN.match(filename)
			if not match:
				continue
			epoch = int(match.group(1))
			xml_path = os.path.join(method_path, filename)
			epochs[epoch] = iter_tripinfo_metrics(xml_path)

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
		plt.title(f"Average {metric} per epoch (cars only)")
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


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	results_dir = os.path.join(base_dir, RESULTS_DIR)
	plots_dir = os.path.join(base_dir, PLOTS_DIR)

	methods = collect_results(results_dir)
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
