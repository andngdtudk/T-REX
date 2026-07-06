"""
Analyze logged car/bike/ped pressure to derive sane W_BIKE / W_PED weights
and a PED_NORM cap for the mplight_mm reward.

Usage:
    python analyze_pressures.py pressure_log.csv [--mode parity|equal_thirds]

Expects columns: step, signal_id, car_pressure, bike_pressure, ped_pressure
NOTE: the 'ped_pressure' column is the RAW (unclipped) value -- the clip is
applied only inside the reward function itself, never written back to the log.

TWO WEIGHTING MODES:
  parity       (default) -- match each mode's p90 contribution to car's p90.
                            car is the reference; bike and ped are scaled to it.
  equal_thirds            -- give each mode an equal 1/3 share of the total
                            reward signal at typical (p90) conditions.
                            Equivalent to IDQN's equal-weight treatment of
                            car/bike/ped signals.
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
    """
    capped = df["ped_pressure"].clip(upper=ped_norm)
    pct_capped = (df["ped_pressure"] > ped_norm).mean() * 100
    car_p90 = df["car_pressure"].abs().quantile(0.90)
    bike_contrib_p90 = (w_bike * df["bike_pressure"].abs()).quantile(0.90)
    ped_contrib_p90 = (w_ped * capped).quantile(0.90)
    total_p90 = car_p90 + bike_contrib_p90 + ped_contrib_p90

    print(f"\n=== Clip check (matches reward fn): "
          f"PED_NORM={ped_norm:.3f}, W_BIKE={w_bike:.4f}, W_PED={w_ped:.4f} ===")
    print(f"  % of steps capped                  : {pct_capped:.2f}%")
    print(f"  typical car contribution (p90)      : {car_p90:.3f}  "
          f"({100*car_p90/total_p90:.1f}% of total)")
    print(f"  typical W_BIKE*bike contribution(p90): {bike_contrib_p90:.3f}  "
          f"({100*bike_contrib_p90/total_p90:.1f}% of total)")
    print(f"  typical W_PED*ped contribution (p90) : {ped_contrib_p90:.3f}  "
          f"({100*ped_contrib_p90/total_p90:.1f}% of total)")
    print(f"  total typical reward magnitude (p90) : {total_p90:.3f}")
    print(f"  max possible W_PED*ped               : {w_ped * ped_norm:.3f}")


def suggest_weights_parity(car_stats, bike_stats, ped_stats, ped_norm, df):
    """Original mode: match bike and ped p90 contributions to car's p90.
    Car is the reference; bike and ped are scaled up/down to match it."""
    print("\n=== Parity mode: match bike/ped p90 to car p90 ===")
    w_bike = (car_stats["p90_abs"] / bike_stats["p90_abs"]
              if bike_stats["p90_abs"] > 0 else 1.0)
    print(f"  W_BIKE ~= {w_bike:.4f}  (matches bike p90 to car p90)")

    car_bike_p90 = (df["car_pressure"] + w_bike * df["bike_pressure"]).abs().quantile(0.90)
    ped_capped_p90 = df["ped_pressure"].clip(upper=ped_norm).quantile(0.90)
    w_ped = car_bike_p90 / ped_capped_p90 if ped_capped_p90 > 0 else 1.0
    print(f"  W_PED  ~= {w_ped:.4f}  (matches ped p90 to combined car+bike p90)")
    return w_bike, w_ped


def suggest_weights_equal_thirds(car_stats, bike_stats, ped_stats, ped_norm, df):
    """Equal-thirds mode: weight each mode so it contributes exactly 1/3 of
    the total reward signal at typical (p90) conditions.

    Solved simultaneously:
        W_BIKE * bike_p90 = car_p90                    ... (bike share == car share)
        W_PED  * ped_capped_p90 = car_p90              ... (ped share == car share)
    which gives equal thirds by construction since all three are equal.

    This is equivalent to IDQN's approach of treating car/bike/ped signals
    with equal importance, just applied to the pressure-based reward instead
    of a delta-wait reward.

    NOTE: 'equal thirds' refers to equal WEIGHTED PRESSURE contributions at
    p90 conditions, not equal raw pressure values. The weights compensate for
    each mode's different natural pressure scale so that, in a typical step,
    none of the three dominates the reward signal.
    """
    print("\n=== Equal-thirds mode: each mode contributes 1/3 of total reward signal ===")
    car_p90 = car_stats["p90_abs"]

    # Step 1: W_BIKE so that W_BIKE * bike_p90 == car_p90
    w_bike = (car_p90 / bike_stats["p90_abs"]
              if bike_stats["p90_abs"] > 0 else 1.0)
    print(f"  W_BIKE ~= {w_bike:.4f}  "
          f"(W_BIKE * bike_p90 [{bike_stats['p90_abs']:.3f}] = car_p90 [{car_p90:.3f}])")

    # Step 2: W_PED so that W_PED * ped_capped_p90 == car_p90
    ped_capped_p90 = df["ped_pressure"].clip(upper=ped_norm).quantile(0.90)
    w_ped = (car_p90 / ped_capped_p90
             if ped_capped_p90 > 0 else 1.0)
    print(f"  W_PED  ~= {w_ped:.4f}  "
          f"(W_PED * ped_capped_p90 [{ped_capped_p90:.3f}] = car_p90 [{car_p90:.3f}])")

    target_share = 100 / 3
    print(f"  Target: each mode ~{target_share:.1f}% of total reward signal at p90 conditions")
    return w_bike, w_ped


def main(path, mode="parity"):
    df = pd.read_csv(path)
    required = {"car_pressure", "bike_pressure", "ped_pressure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Log is missing columns: {missing}")

    print(f"Loaded {len(df)} rows from {path}")
    print(f"Signals: {df['signal_id'].nunique()}  Steps: {df['step'].nunique()}")
    if df['step'].nunique() <= 1:
        print("  WARNING: only 1 unique step value logged -- fix the step "
              "counter before trusting percentiles below.")

    car_stats = summarize("car_pressure", df["car_pressure"])
    bike_stats = summarize("bike_pressure", df["bike_pressure"])
    ped_stats = summarize("ped_pressure", df["ped_pressure"])

    print(f"\n=== PED_NORM from raw ped distribution ===")
    ped_norm = ped_stats["p99_abs"]
    print(f"  PED_NORM ~= {ped_norm:.3f}  (ped p99 -- caps rare crowd spikes)")

    if mode == "equal_thirds":
        w_bike, w_ped = suggest_weights_equal_thirds(
            car_stats, bike_stats, ped_stats, ped_norm, df)
    else:
        w_bike, w_ped = suggest_weights_parity(
            car_stats, bike_stats, ped_stats, ped_norm, df)

    print("\n=== Verification with derived settings ===")
    check_clip_settings(df, ped_norm=ped_norm, w_bike=w_bike, w_ped=w_ped)

    print("\nNote: these are starting points for a sweep, not final values. "
          "Re-check this distribution after training under the new weights -- "
          "policy changes will shift the pressure distributions themselves.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pressures.py <pressure_log.csv> [--mode parity|equal_thirds]")
        sys.exit(1)
    csv_path = sys.argv[1]
    run_mode = "parity"
    if len(sys.argv) >= 4 and sys.argv[2] == "--mode":
        run_mode = sys.argv[3]
    if run_mode not in ("parity", "equal_thirds"):
        print(f"Unknown mode '{run_mode}'. Use 'parity' or 'equal_thirds'.")
        sys.exit(1)
    main(csv_path, mode=run_mode)