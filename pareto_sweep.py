"""
Compute and visualize the Pareto frontier across a (W_BIKE, W_PED) sweep.

Expects a CSV with one row per completed training run:
    w_bike, w_ped, car_wait, bike_wait, ped_wait
where car_wait/bike_wait/ped_wait are your chosen summary stat (e.g. mean
wait over the last N epochs) -- lower is better for all three.

Usage:
    python pareto_sweep.py sweep_results.csv

Outputs:
    - prints the Pareto-optimal runs (non-dominated set)
    - saves a 3D scatter (pareto_3d.png) and three 2D pairwise projections
      (pareto_2d_car_bike.png, pareto_2d_car_ped.png, pareto_2d_bike_ped.png)
      highlighting Pareto-optimal points
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import scienceplots

plt.style.use(['science'])

# Distinct markers to cycle through for Pareto-optimal points (no edge color,
# each point gets its own color from a qualitative colormap instead).
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']

AXIS_LABELS = {
    "car_wait": "Car wait",
    "bike_wait": "Bike wait",
    "ped_wait": "Pedestrian wait",
}


def is_dominated(row, others, cols):
    """A row is dominated if some other row is <= on all objectives and
    strictly < on at least one (lower = better for all cols here)."""
    for _, other in others.iterrows():
        if all(other[c] <= row[c] for c in cols) and any(other[c] < row[c] for c in cols):
            return True
    return False


def compute_pareto(df, cols):
    mask = []
    for idx, row in df.iterrows():
        others = df.drop(idx)
        mask.append(not is_dominated(row, others, cols))
    return df[mask].copy()


def pareto_style(pareto_df):
    """Assign each Pareto-optimal point a distinct color + marker, and build
    a clean legend label carrying the (w_bike, w_ped) info."""
    n = len(pareto_df)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n)]
    markers = [MARKERS[i % len(MARKERS)] for i in range(n)]
    labels = [f"WB={r.w_bike:.2g}, WP={r.w_ped:.2g}" for r in pareto_df.itertuples()]
    return colors, markers, labels


def main(path):
    df = pd.read_csv(path)
    required = {"w_bike", "w_ped", "car_wait", "bike_wait", "ped_wait"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    cols = ["car_wait", "bike_wait", "ped_wait"]
    print(f"Loaded {len(df)} runs from {path}")

    pareto_df = compute_pareto(df, cols)
    pareto_df = pareto_df.sort_values("car_wait")
    print(f"\n{len(pareto_df)} Pareto-optimal run(s) out of {len(df)}:\n")
    print(pareto_df[["w_bike", "w_ped"] + cols].to_string(index=False))

    dominated_df = df.drop(pareto_df.index)
    colors, markers, labels = pareto_style(pareto_df)

    # --- 3D scatter ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(dominated_df["car_wait"], dominated_df["bike_wait"], dominated_df["ped_wait"],
               c="lightgray", label="Dominated", s=40)
    for (_, r), color, marker, label in zip(pareto_df.iterrows(), colors, markers, labels):
        ax.scatter(r["car_wait"], r["bike_wait"], r["ped_wait"],
                   c=[color], marker=marker, s=70, label=label)
    ax.set_xlabel(f"{AXIS_LABELS['car_wait']} [s]")
    ax.set_ylabel(f"{AXIS_LABELS['bike_wait']} [s]")
    ax.set_zlabel(f"{AXIS_LABELS['ped_wait']} [s]")
    ax.set_title("Pareto frontier across W_BIKE / W_PED sweep")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1.0),
              markerscale=0.7, labelspacing=1.2, handletextpad=0.8, borderpad=1.0)
    plt.tight_layout()
    plt.savefig("pareto_3d.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2D pairwise projections ---
    pairs = [("car_wait", "bike_wait"), ("car_wait", "ped_wait"), ("bike_wait", "ped_wait")]
    for x, y in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(dominated_df[x], dominated_df[y], c="lightgray", label="Dominated", s=40)
        for (_, r), color, marker, label in zip(pareto_df.iterrows(), colors, markers, labels):
            ax.scatter(r[x], r[y], c=[color], marker=marker, s=70, label=label)
        ax.set_xlabel(f"{AXIS_LABELS[x]} [s]")
        ax.set_ylabel(f"{AXIS_LABELS[y]} [s]")
        ax.set_title(f"{AXIS_LABELS[x]} vs {AXIS_LABELS[y]}")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  markerscale=0.7, labelspacing=1.2, handletextpad=0.8, borderpad=1.0)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"pareto_2d_{x.split('_')[0]}_{y.split('_')[0]}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    print("\nSaved: pareto_3d.png, pareto_2d_car_bike.png, "
          "pareto_2d_car_ped.png, pareto_2d_bike_ped.png")
    print("\nNote: points shown as 'Pareto-optimal' are non-dominated across "
          "ALL THREE objectives jointly. A point can look beaten in one 2D "
          "projection (e.g. another point has lower car+bike) but still be "
          "on the true 3D frontier because it wins on the third axis (ped).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pareto_sweep.py <sweep_results.csv>")
        sys.exit(1)
    main(sys.argv[1])