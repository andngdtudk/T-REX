"""
Analyze logged car/bike/ped pressure to derive sane W_BIKE / W_PED weights
and a PED_NORM cap for the mplight_mm reward.

Usage:
    python analyze_pressures.py pressure_log.csv

Expects columns: step, signal_id, car_pressure, bike_pressure, ped_pressure
NOTE: per the actual _log_pressures() call site, the 'ped_pressure' column
in the CSV is the RAW (unclipped) pedestrian pressure -- the clip is applied
only inside the reward function itself (min(ped_pressure_raw, PED_NORM)),
never written back to the log. That's intentional: it lets this script see
the true unclipped distribution.
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


def check_clip_settings(df, ped_norm, w_bike, w_ped):
    """Matches the ACTUAL reward implementation:
        ped_pressure = min(ped_pressure_raw, PED_NORM)
        reward = -(car_pressure + W_BIKE*bike_pressure + W_PED*ped_pressure)
    Single-knob one-sided cap at PED_NORM (ped_pressure_raw is non-negative
    by construction, so there's no lower tail to worry about).
    """
    capped = df["ped_pressure"].clip(upper=ped_norm)
    pct_capped = (df["ped_pressure"] > ped_norm).mean() * 100
    car_bike = (df["car_pressure"] + w_bike * df["bike_pressure"]).abs()

    ped_contrib_p90 = (w_ped * capped).quantile(0.90)
    car_bike_p90 = car_bike.quantile(0.90)

    print(f"\n=== Clip check (matches reward fn): "
          f"PED_NORM={ped_norm}, W_BIKE={w_bike}, W_PED={w_ped} ===")
    print(f"  % of steps capped            : {pct_capped:.2f}%")
    print(f"  typical |car+bike| (p90)     : {car_bike_p90:.3f}")
    print(f"  typical W_PED*ped_capped(p90): {ped_contrib_p90:.3f}")
    print(f"  max possible W_PED*ped       : {w_ped * ped_norm:.3f}")
    if car_bike_p90 > 0:
        print(f"  ratio ped/combined (target ~1.0): {ped_contrib_p90 / car_bike_p90:.3f}")


def suggest_w_ped_against_combined(df, w_bike, ped_norm):
    """W_PED needs to hold parity against car_pressure + W_BIKE*bike_pressure
    COMBINED (the actual competing term in the summed reward), not against
    car_pressure alone."""
    car_bike_p90 = (df["car_pressure"] + w_bike * df["bike_pressure"]).abs().quantile(0.90)
    ped_capped_p90 = df["ped_pressure"].clip(upper=ped_norm).quantile(0.90)
    w_ped = car_bike_p90 / ped_capped_p90 if ped_capped_p90 > 0 else 1.0
    print(f"\n=== W_PED suggestion against COMBINED car+bike (W_BIKE={w_bike}) ===")
    print(f"  p90(|car + {w_bike}*bike|)            : {car_bike_p90:.3f}")
    print(f"  p90(ped_pressure capped at {ped_norm}) : {ped_capped_p90:.3f}")
    print(f"  W_PED ~= {w_ped:.4f}")
    return w_ped


def main(path):
    df = pd.read_csv(path)
    required = {"car_pressure", "bike_pressure", "ped_pressure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Log is missing columns: {missing}")

    print(f"Loaded {len(df)} rows from {path}")
    print(f"Signals: {df['signal_id'].nunique()}  Steps: {df['step'].nunique()}")
    if df['step'].nunique() <= 1:
        print("  WARNING: only 1 unique step value logged -- this almost "
              "certainly means the 'step' argument passed into the reward "
              "fn / _log_pressures isn't a real incrementing counter, or "
              "the log file has stale data from a previous run mixed in. "
              "Fix this before trusting percentiles below.")

    car_stats = summarize("car_pressure", df["car_pressure"])
    bike_stats = summarize("bike_pressure", df["bike_pressure"])
    ped_stats = summarize("ped_pressure", df["ped_pressure"])

    print("\n=== Step 1: W_BIKE against car_pressure alone ===")
    w_bike_scale = (car_stats["p90_abs"] / bike_stats["p90_abs"]
                     if bike_stats["p90_abs"] > 0 else 1.0)
    print(f"  W_BIKE ~= {w_bike_scale:.4f}  (matches bike p90 to car p90)")

    print("\n=== Step 2: PED_NORM from raw ped distribution ===")
    ped_norm = ped_stats["p99_abs"]
    print(f"  PED_NORM ~= {ped_norm:.3f}  (ped p99 -- caps rare crowd spikes, "
          f"leaves typical variation untouched)")

    print("\n=== Step 3: W_PED against car+bike COMBINED (uses W_BIKE from step 1) ===")
    w_ped = suggest_w_ped_against_combined(df, w_bike=w_bike_scale, ped_norm=ped_norm)

    print("\n=== Verification with derived settings ===")
    check_clip_settings(df, ped_norm=ped_norm, w_bike=w_bike_scale, w_ped=w_ped)

    print("\nNote: these are starting points for a sweep, not final values. "
          "Re-check this distribution after training under the new weights -- "
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