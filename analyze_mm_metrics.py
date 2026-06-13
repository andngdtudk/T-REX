"""
analyze_mm_metrics.py
---------------------
Standalone analysis script for MultimodalLogger output.

Run after training:
    python analyze_mm_metrics.py --log_dir results/IDQN_MM2-tr0-kbh_j1_442m/

Produces:
  - mm_analysis_report.txt   (printed summary)
  - mm_plots.png             (4-panel figure)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — skipping plots.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_episode_df(log_dir: str) -> pd.DataFrame:
    path = os.path.join(log_dir, "mm_episode_metrics.csv")
    if not os.path.exists(path):
        sys.exit(f"[error] File not found: {path}\nRun training first.")
    return pd.read_csv(path)


def load_step_df(log_dir: str) -> pd.DataFrame:
    path = os.path.join(log_dir, "mm_step_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def smooth(series, window=5):
    return series.rolling(window, min_periods=1, center=True).mean()


def print_report(ep: pd.DataFrame):
    print("=" * 60)
    print("  IDQN_MM2 — Multimodal Training Report")
    print("=" * 60)

    n = len(ep)
    first10 = ep.head(10)
    last10  = ep.tail(10)

    def pct_change(col):
        a = first10[col].mean()
        b = last10[col].mean()
        if a == 0:
            return float("nan")
        return (b - a) / abs(a) * 100

    print(f"\nEpisodes recorded : {n}")
    print(f"{'Metric':<25} {'First-10 mean':>14} {'Last-10 mean':>14} {'Δ%':>8}")
    print("-" * 65)
    for col, label in [
        ("car_wait_total",   "Car wait (s·steps)"),
        ("bike_wait_total",  "Bike wait (s·steps)"),
        ("ped_wait_total",   "Ped wait (s·steps)"),
        ("car_queue_total",  "Car queue (veh·steps)"),
        ("bike_queue_total", "Bike queue (veh·steps)"),
        ("reward_mean",      "Mean reward"),
        ("mean_loss",        "Mean loss"),
        ("mean_q",           "Mean Q-value"),
    ]:
        if col not in ep.columns:
            continue
        a = first10[col].mean()
        b = last10[col].mean()
        pct = pct_change(col)
        sign = "↓" if pct < 0 else "↑"
        print(f"{label:<25} {a:>14.3f} {b:>14.3f} {sign}{abs(pct):>6.1f}%")

    # Mode dominance check
    print("\n--- Mode fraction (last 10 episodes) ---")
    bfrac = last10["bike_wait_frac"].mean() * 100
    pfrac = last10["ped_wait_frac"].mean() * 100
    cfrac = 100 - bfrac - pfrac
    print(f"  Car  : {cfrac:.1f}%")
    print(f"  Bike : {bfrac:.1f}%")
    print(f"  Ped  : {pfrac:.1f}%")

    if bfrac + pfrac > 60:
        print("\n  [!] Bikes + peds dominate >60% of total wait.")
        print("      Consider reducing bike_wait_weight / ped_wait_weight in the reward config.")
    elif bfrac + pfrac < 5:
        print("\n  [!] Bike/ped contribution is negligible (<5%).")
        print("      Check that bike_total_wait and ped_total_wait are non-zero in observations.")

    # Reward saturation check
    rmin = ep["reward_min"].mean()
    rmax = ep["reward_max"].mean()
    print(f"\n--- Reward range (episode mean) ---")
    print(f"  Min : {rmin:.4f}   Max : {rmax:.4f}")
    if abs(rmin) > 3.5 or abs(rmax) > 3.5:
        print("  [!] Reward frequently hits clip limits (±4).")
        print("      norm_wait may be too small → consider increasing it.")

    # Epsilon check
    print(f"\n--- Exploration (epsilon) ---")
    final_eps = ep["epsilon"].iloc[-1] if "epsilon" in ep.columns else float("nan")
    print(f"  Final epsilon : {final_eps:.4f}")
    if final_eps > 0.15:
        print("  [!] Epsilon still high at end — consider more episodes or faster decay.")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(ep: pd.DataFrame, step: pd.DataFrame, out_path: str):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IDQN_MM2 — Multimodal Training Diagnostics", fontsize=13, fontweight="bold")

    x = ep["episode"]

    # Panel 1 — per-mode wait over episodes
    ax = axes[0, 0]
    for col, label, color in [
        ("car_wait_total",  "Car",  "steelblue"),
        ("bike_wait_total", "Bike", "darkorange"),
        ("ped_wait_total",  "Ped",  "seagreen"),
    ]:
        if col in ep.columns:
            ax.plot(x, smooth(ep[col]), label=label, color=color, linewidth=1.5)
            ax.fill_between(x, ep[col], alpha=0.07, color=color)
    ax.set_title("Total wait per mode (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accumulated wait (s·steps)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2 — mode fraction over time
    ax = axes[0, 1]
    total = ep["car_wait_total"] + ep["bike_wait_total"] + ep["ped_wait_total"]
    total = total.replace(0, np.nan)
    for col, label, color in [
        ("car_wait_total",  "Car",  "steelblue"),
        ("bike_wait_total", "Bike", "darkorange"),
        ("ped_wait_total",  "Ped",  "seagreen"),
    ]:
        if col in ep.columns:
            ax.plot(x, smooth(ep[col] / total * 100), label=label, color=color, linewidth=1.5)
    ax.set_title("Wait fraction per mode (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3 — reward mean + std band
    ax = axes[0, 2]
    rm = smooth(ep["reward_mean"])
    rs = smooth(ep["reward_std"])
    ax.plot(x, rm, color="purple", linewidth=1.5, label="Mean reward")
    ax.fill_between(x, rm - rs, rm + rs, alpha=0.15, color="purple", label="±1 std")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Reward signal")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (normalised, clipped)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4 — Q-value and loss
    ax = axes[1, 0]
    ax2 = ax.twinx()
    if "mean_q" in ep.columns:
        ax.plot(x, smooth(ep["mean_q"]), color="teal", linewidth=1.5, label="Mean Q")
        ax.set_ylabel("Mean Q", color="teal")
        ax.tick_params(axis="y", labelcolor="teal")
    if "mean_loss" in ep.columns:
        ax2.plot(x, smooth(ep["mean_loss"]), color="crimson", linewidth=1, linestyle="--", label="Loss")
        ax2.set_ylabel("Loss", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
    ax.set_title("Q-value & Loss")
    ax.set_xlabel("Episode")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5 — queue breakdown
    ax = axes[1, 1]
    if "car_queue_total" in ep.columns:
        ax.plot(x, smooth(ep["car_queue_total"]),  label="Car queue",  color="steelblue",  linewidth=1.5)
    if "bike_queue_total" in ep.columns:
        ax.plot(x, smooth(ep["bike_queue_total"]), label="Bike queue", color="darkorange", linewidth=1.5)
    ax.set_title("Queue length per mode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accumulated queue (veh·steps)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 6 — step-level reward distribution (last 10 episodes, box-per-10ep bucket)
    ax = axes[1, 2]
    if step is not None and "reward_mean" in step.columns:
        max_ep = step["episode"].max()
        bucket_size = max(1, max_ep // 10)
        step["ep_bucket"] = ((step["episode"] - 1) // bucket_size) * bucket_size + 1
        buckets = sorted(step["ep_bucket"].unique())
        data_to_plot = [step.loc[step["ep_bucket"] == b, "reward_mean"].values for b in buckets]
        ax.boxplot(data_to_plot, labels=[str(b) for b in buckets], showfliers=False)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title("Step reward distribution (buckets)")
        ax.set_xlabel("Episode bucket start")
        ax.set_ylabel("Step reward (mean over signals)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "step CSV not found", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title("Step reward distribution")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plots] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse IDQN_MM2 multimodal training logs.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing mm_episode_metrics.csv (and optionally mm_step_metrics.csv)")
    args = parser.parse_args()

    ep   = load_episode_df(args.log_dir)
    step = load_step_df(args.log_dir)

    print_report(ep)

    plot_path = os.path.join(args.log_dir, "mm_plots.png")
    make_plots(ep, step, plot_path)


if __name__ == "__main__":
    main()
