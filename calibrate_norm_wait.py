"""
calibrate_norm_wait.py
----------------------
Reads mm_step_metrics.csv from a completed (or partial) run and computes
a recommended norm_wait for wait_multimodal_delta_sclip.

The key insight: norm_wait should equal the typical *absolute* per-step
delta in combined_wait, so the reward signal spans roughly [-1, +1]
before clipping.  clip_wait can then be set to 2-3 to allow a useful
dynamic range without throwing away information.

Usage:
    python calibrate_norm_wait.py --log_dir results/IDQN_MM2-tr0-kbh_j1_442m-0-drq_mm2_delta-wait_multimodal_delta_sclip/

Prints recommended mdp_configs values and writes calibration_report.txt.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    # Optional weight overrides — must match what you intend to use
    parser.add_argument("--car_w",  type=float, default=1.0)
    parser.add_argument("--bike_w", type=float, default=1.0)
    parser.add_argument("--ped_w",  type=float, default=1.0)
    args = parser.parse_args()

    step_path = os.path.join(args.log_dir, "mm_step_metrics.csv")
    if not os.path.exists(step_path):
        sys.exit(f"[error] {step_path} not found.")

    df = pd.read_csv(step_path)

    # Reconstruct combined wait the same way the reward function does
    df["combined_wait"] = (
        args.car_w  * df["car_wait"] +
        args.bike_w * df["bike_wait"] +
        args.ped_w  * df["ped_wait"]
    )

    # Per-episode delta series (step-to-step difference within episode)
    deltas = []
    for ep, group in df.groupby("episode"):
        d = group["combined_wait"].diff().dropna().abs()
        deltas.extend(d.tolist())

    deltas = np.array(deltas)
    deltas = deltas[np.isfinite(deltas)]

    p50  = np.percentile(deltas, 50)
    p75  = np.percentile(deltas, 75)
    p90  = np.percentile(deltas, 90)
    p95  = np.percentile(deltas, 95)
    p99  = np.percentile(deltas, 99)
    mean = deltas.mean()
    std  = deltas.std()

    # Recommendations
    # norm_wait ≈ p75 so ~75% of steps have |reward| < 1 before clipping
    # clip_wait = 3.0 gives headroom for outliers without excessive saturation
    recommended_norm  = round(float(p75), 1)
    recommended_clip  = 3.0

    # Bike-weight adjustment: if bikes dominate, down-weight them so
    # each mode contributes roughly equally after weighting.
    # Target: car_mean_wait * car_w ≈ bike_mean_wait * bike_w
    car_mean  = df["car_wait"].mean()
    bike_mean = df["bike_wait"].mean()
    ped_mean  = df["ped_wait"].mean()

    # Normalise weights so car_wait_weight stays at 1.0
    if bike_mean > 0 and car_mean > 0:
        bike_w_balanced = round(car_mean / bike_mean, 4)
    else:
        bike_w_balanced = 1.0
    if ped_mean > 0 and car_mean > 0:
        ped_w_balanced = round(car_mean / ped_mean, 4)
    else:
        ped_w_balanced = 1.0

    lines = []
    lines.append("=" * 60)
    lines.append("  norm_wait / clip_wait calibration report")
    lines.append("=" * 60)
    lines.append(f"\nDataset : {len(deltas):,} step-deltas across {df['episode'].nunique()} episodes")
    lines.append(f"\nPer-step |Δ combined_wait| distribution:")
    lines.append(f"  mean : {mean:>12.1f}")
    lines.append(f"  std  : {std:>12.1f}")
    lines.append(f"  p50  : {p50:>12.1f}")
    lines.append(f"  p75  : {p75:>12.1f}  ← recommended norm_wait")
    lines.append(f"  p90  : {p90:>12.1f}")
    lines.append(f"  p95  : {p95:>12.1f}")
    lines.append(f"  p99  : {p99:>12.1f}")
    lines.append(f"\nMean accumulated wait per step:")
    lines.append(f"  car  : {car_mean:>12.1f}")
    lines.append(f"  bike : {bike_mean:>12.1f}  ({bike_mean/max(car_mean,1):.1f}x car)")
    lines.append(f"  ped  : {ped_mean:>12.1f}  ({ped_mean/max(car_mean,1):.2f}x car)")
    lines.append(f"\n--- Recommended mdp_configs entry ---")
    lines.append(f"""
'IDQN_MM2': {{
    # --- calibrated from run data ---
    'norm_wait'        : {recommended_norm},
    'clip_wait'        : {recommended_clip},

    # Option A: equal weights, larger norm  (simpler, try first)
    'car_wait_weight'  : 1.0,
    'bike_wait_weight' : 1.0,
    'ped_wait_weight'  : 1.0,

    # Option B: balanced weights so each mode contributes equally to reward
    # 'car_wait_weight'  : 1.0,
    # 'bike_wait_weight' : {bike_w_balanced},   # down-weights bike by {1/max(bike_w_balanced,1e-9):.1f}x
    # 'ped_wait_weight'  : {ped_w_balanced},
}},""")
    lines.append(f"""
--- What this fixes ---
Current norm_wait=224 → typical delta ({p75:.0f}) / 224 = {p75/224:.1f}x clip limit.
Every reward is clipped to ±{recommended_clip}; the agent sees only a binary signal.

New norm_wait={recommended_norm} → typical delta normalises to ~1.0 before clipping.
clip_wait={recommended_clip} keeps the ±4 headroom for outlier steps.

If you want to reduce bike dominance, use Option B weights.
Bike currently contributes {bike_mean/(car_mean+bike_mean+ped_mean)*100:.0f}% of combined wait;
balanced weights bring it to ~33%.
""")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(args.log_dir, "calibration_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
