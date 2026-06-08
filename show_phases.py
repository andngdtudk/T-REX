import argparse
import os
import re

import matplotlib.pyplot as plt


def parse_dict(dict_str):
    cleaned = dict_str.replace("np.float32(", "").replace(")", "")
    pairs = re.findall(r"'([^']+)'\s*:\s*([^,}]+)", cleaned)
    data = {}
    for key, value in pairs:
        value = value.strip()
        try:
            if any(ch in value for ch in [".", "e", "E"]):
                data[key] = float(value)
            else:
                data[key] = int(value)
        except ValueError:
            try:
                data[key] = float(value)
            except ValueError:
                continue
    return data


def parse_line(line, dict_index):
    dicts = re.findall(r"\{[^}]*\}", line)
    if len(dicts) <= dict_index:
        return None
    time_str = line.split(",", 1)[0].strip()
    try:
        time_value = float(time_str)
    except ValueError:
        return None
    data = parse_dict(dicts[dict_index])
    return time_value, data


def load_series(csv_path, dict_index, junction):
    times = []
    values = []
    chosen_junction = junction

    with open(csv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = parse_line(line, dict_index)
            if not parsed:
                continue
            time_value, data = parsed
            if not data:
                continue
            if chosen_junction is None:
                chosen_junction = next(iter(data.keys()))
            if chosen_junction not in data:
                continue
            times.append(time_value)
            values.append(data[chosen_junction])

    return times, values, chosen_junction


def plot_series(times, values, junction, output_path, show_plot):
    plt.figure(figsize=(12, 4))
    unique_values = sorted(set(values))
    palette = plt.get_cmap("tab20")
    colors = {val: palette(i % palette.N) for i, val in enumerate(unique_values)}
    point_colors = [colors[val] for val in values]

    plt.scatter(times, values, s=20, c=point_colors, zorder=3)
    plt.xlabel("Time")
    plt.ylabel("Phase")
    plt.title(f"Phase choices over time ({junction})")
    plt.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=colors[val], markersize=6,
                          label=str(val))
               for val in unique_values]
    plt.legend(handles=handles, title="Phase", ncol=4, fontsize=8,
               frameon=False, loc="lower right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot discrete option choices over time.")
    parser.add_argument("csv_path", help="Path to metrics_*.csv")
    parser.add_argument("--dict-index", type=int, default=1,
                        help="0-based index among dict columns to use (default: 1 = third column).")
    parser.add_argument("--junction", default=None, help="Junction key to plot (default: first key found).")
    parser.add_argument("--output", default=None, help="Output image path (default: show_phases.png next to CSV).")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(args.csv_path))
        output_path = os.path.join(base_dir, "show_phases.png")

    times, values, junction = load_series(args.csv_path, args.dict_index, args.junction)
    if not times:
        raise SystemExit("No data points found. Check --dict-index or --junction.")

    plot_series(times, values, junction, output_path, args.show)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
