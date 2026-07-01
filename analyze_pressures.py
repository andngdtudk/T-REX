"""
Analyze logged car/bike/ped pressure to derive sane W_BIKE / W_PED weights
and a PED_NORM clip value for mplight_mm reward.

Usage:
    python analyze_pressures.py pressure_log.csv

Expects columns: step, signal_id, car_pressure, bike_pressure, ped_pressure_raw
(see _log_pressures() snippet added to the reward fn).
"""
import sys
import numpy as np
import pandas as pd


def summarize(name, series):
    abs_series = series.abs()
    print(f"\n--- {name} ---")
    print(f"  mean        : {series.mean():.3f}")
    print(f"  std         : {series.std():.3f}")
    print(f"  |mean|      : {abs_series.mean():.3f}")
    for p in (50, 75, 90, 95, 99, 99.9):
        print(f"  |p{p:>5}|    : {np.percentile(abs_series, p):.3f}")
    print(f"  max(|x|)    : {abs_series.max():.3f}")
    return {
        "mean_abs": abs_series.mean(),
        "p90_abs": np.percentile(abs_series, 90),
        "p99_abs": np.percentile(abs_series, 99),
        "max_abs": abs_series.max(),
    }


def main(path):
    df = pd.read_csv(path)
    required = {"car_pressure", "bike_pressure", "ped_pressure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Log is missing columns: {missing}")

    print(f"Loaded {len(df)} rows from {path}")
    print(f"Signals: {df['signal_id'].nunique()}  Steps: {df['step'].nunique()}")

    car_stats = summarize("car_pressure", df["car_pressure"])
    bike_stats = summarize("bike_pressure", df["bike_pressure"])
    ped_stats = summarize("ped_pressure", df["ped_pressure"])

    # --- Suggested normalization ---
    # Strategy: match the *typical* (p90, robust-to-outliers) magnitude of
    # ped_pressure to car_pressure's typical magnitude, then weight bike
    # similarly relative to car. Using p90 instead of max avoids letting a
    # single crowd-crossing event set the scale for the entire training run.
    print("\n=== Suggested starting weights ===")

    if ped_stats["p90_abs"] > 0:
        w_ped_scale = car_stats["p90_abs"] / ped_stats["p90_abs"]
    else:
        w_ped_scale = 1.0
    if bike_stats["p90_abs"] > 0:
        w_bike_scale = car_stats["p90_abs"] / bike_stats["p90_abs"]
    else:
        w_bike_scale = 1.0

    print(f"  W_BIKE ~= {w_bike_scale:.4f}  (matches bike p90 |pressure| to car p90)")
    print(f"  W_PED  ~= {w_ped_scale:.4f}  (matches ped p90 |pressure| to car p90)")

    # Clip recommendation: cap ped_pressure at its own p99, so the rare
    # crowd-crossing event can't dominate any single reward, but everyday
    # variation up to "busy crossing" is still felt.
    ped_p99 = ped_stats["p99_abs"]
    print(f"\n  PED_NORM (clip pre-weighting) ~= {ped_p99:.3f}  (ped p99 |pressure|)")
    print(f"  -> after clip+scale, ped contribution to reward capped at "
          f"~{ped_p99 * w_ped_scale:.3f} in magnitude, comparable to car's "
          f"typical (p90) contribution of ~{car_stats['p90_abs']:.3f}")

    print("\nNote: these are starting points for a sweep, not final values. "
          "Re-check this distribution after training under the new weights — "
          "policy changes will shift the pressure distributions themselves.")

    df = pd.read_csv("pressure_log.csv")
    _check_clip_settings(df, ped_norm=22, ped_clip=4, w_bike=2.2, w_ped=3)


def _check_clip_settings(df, ped_norm, ped_clip, w_bike, w_ped):
    normed = df["ped_pressure"] / ped_norm
    clipped = normed.clip(-ped_clip, ped_clip)
    pct_clipped = (normed.abs() > ped_clip).mean() * 100
    car_bike = (df["car_pressure"] + w_bike * df["bike_pressure"]).abs()

    print(f"\n=== Clip check: PED_NORM={ped_norm}, PED_CLIP={ped_clip}, "
          f"W_BIKE={w_bike}, W_PED={w_ped} ===")
    print(f"  % of steps clipped         : {pct_clipped:.2f}%")
    print(f"  typical |car+bike| (p90)   : {car_bike.quantile(0.90):.3f}")
    print(f"  typical W_PED*|ped| (p90)  : {(w_ped * clipped.abs()).quantile(0.90):.3f}")
    print(f"  max possible W_PED*ped     : {w_ped * ped_clip:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_pressures.py <pressure_log.csv>")
        sys.exit(1)
    main(sys.argv[1])