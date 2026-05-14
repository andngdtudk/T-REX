import os
import re
import math
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOTS_DIR = "plots"
TRIPINFO_PATTERN = re.compile(r"^tripinfo_(\d+)\.xml$")
CAR_PREFIX = "car0_"
METRICS = ("duration", "waitingTime", "stopTime", "timeLoss")
# If True, skip trips that are not finished (arrival == -1 or vaporized == "end").
SKIP_UNFINISHED = True


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


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	results_dir = os.path.join(base_dir, RESULTS_DIR)
	plots_dir = os.path.join(base_dir, PLOTS_DIR)

	methods = collect_results(results_dir)
	if not methods:
		print(f"No tripinfo files found in {results_dir}.")
		return

	plot_metrics(methods, plots_dir)
	print(f"Plots saved to {plots_dir}.")


if __name__ == "__main__":
	main()
