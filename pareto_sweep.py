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

    # --- 3D scatter ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(dominated_df["car_wait"], dominated_df["bike_wait"], dominated_df["ped_wait"],
               c="lightgray", label="dominated", s=40)
    ax.scatter(pareto_df["car_wait"], pareto_df["bike_wait"], pareto_df["ped_wait"],
               c="crimson", label="Pareto-optimal", s=70, edgecolor="black")
    for _, r in pareto_df.iterrows():
        ax.text(r["car_wait"], r["bike_wait"], r["ped_wait"],
                 f"  WB={r['w_bike']:.2g},WP={r['w_ped']:.2g}", fontsize=7)
    ax.set_xlabel("car_wait")
    ax.set_ylabel("bike_wait")
    ax.set_zlabel("ped_wait")
    ax.set_title("Pareto frontier across W_BIKE / W_PED sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pareto_3d.png", dpi=150)
    plt.close()

    # --- 2D pairwise projections ---
    pairs = [("car_wait", "bike_wait"), ("car_wait", "ped_wait"), ("bike_wait", "ped_wait")]
    for x, y in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(dominated_df[x], dominated_df[y], c="lightgray", label="dominated", s=40)
        ax.scatter(pareto_df[x], pareto_df[y], c="crimson", label="Pareto-optimal (3-obj)",
                   s=70, edgecolor="black")
        for _, r in pareto_df.iterrows():
            ax.annotate(f"WB={r['w_bike']:.2g},WP={r['w_ped']:.2g}",
                        (r[x], r[y]), fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y} (Pareto pts shown are 3-objective optimal, not 2D-optimal)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"pareto_2d_{x.split('_')[0]}_{y.split('_')[0]}.png"
        plt.savefig(fname, dpi=150)
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