"""
Compile Pareto sweep results into a single CSV for pareto_sweep.py.

Usage:
    python compile_sweep.py <results_dir> [--last-n 20] [--out sweep_data.csv]

For each subfolder in results_dir that contains a *_mm_step_metrics* file:
  - Extracts w_bike and w_ped from the folder name (wb{X} and wp{Y} tokens)
  - Reads the metrics file, takes the last N episodes
  - Averages car_wait, bike_wait, ped_wait over those episodes
  - Writes one row per run to the output CSV

Output CSV columns: w_bike, w_ped, car_wait, bike_wait, ped_wait
Output filename: {MODEL_NAME}_sweep_data.csv (inferred from the common
prefix of all folder names, before the wb token)

Example folder names this handles:
  IDQN_MM2-tr0-wb0.40_wp120.00-kbh_j1_442m-0-drq_mm2_delta-wait_multimodal_delta_sclip
  IDQN_MM2-tr0-wb1.60_wp300.00-kbh_j1_442m-0-drq_mm2_delta-wait_multimodal_delta_sclip
"""

import re
import sys
import argparse
from pathlib import Path

import pandas as pd


# Regex to extract wb and wp values from folder name.
# Handles both underscore and dash separators between wb and wp tokens,
# and float values with any number of decimal places.
WB_WP_RE = re.compile(r'wb(\d+(?:\.\d+)?)[_\-]wp(\d+(?:\.\d+)?)', re.IGNORECASE)


def extract_weights(folder_name: str):
    """Return (w_bike, w_ped) floats from a folder name, or None if not found."""
    m = WB_WP_RE.search(folder_name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return Noneresultsb


def find_metrics_file(folder: Path):
    """Return the first *_mm_step_metrics* CSV file found in folder, or None."""
    candidates = sorted(folder.glob('*_mm_step_metrics*'))
    if not candidates:
        # also try without underscore prefix
        candidates = sorted(folder.glob('*mm_step_metrics*'))
    return candidates[0] if candidates else None


def infer_model_name(folder_names):
    """Infer the model name prefix (everything before the 'wb' token)
    from the list of folder names, using the most common prefix."""
    prefixes = []
    for name in folder_names:
        m = WB_WP_RE.search(name)
        if m:
            # take everything before the wb token, strip trailing separators
            prefix = name[:m.start()].rstrip('-_')
            prefixes.append(prefix)
    if not prefixes:
        return 'sweep'
    # use the most common prefix (should be identical across all runs)
    from collections import Counter
    return Counter(prefixes).most_common(1)[0][0]


def process_run(folder: Path, metrics_file: Path, last_n: int):
    """Read metrics file, return mean of last_n episodes for the three wait cols."""
    try:
        df = pd.read_csv(metrics_file)
    except Exception as e:
        print(f"  [WARN] could not read {metrics_file}: {e}")
        return None

    required = {'episode', 'car_wait', 'bike_wait', 'ped_wait'}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] {metrics_file.name} missing columns: {missing} — skipping")
        return None

    # one row per (episode, agent/decision) — collapse to per-episode means first
    per_episode = (
        df.groupby('episode')[['car_wait', 'bike_wait', 'ped_wait']]
        .mean()
        .reset_index()
        .sort_values('episode')
    )

    if len(per_episode) == 0:
        print(f"  [WARN] {metrics_file.name} has no episode data — skipping")
        return None

    last = per_episode.tail(last_n)
    if len(last) < last_n:
        print(f"  [WARN] {folder.name}: only {len(last)} episodes available "
              f"(wanted {last_n}) — averaging what's there")

    return {
        'car_wait':  last['car_wait'].mean(),
        'bike_wait': last['bike_wait'].mean(),
        'ped_wait':  last['ped_wait'].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description='Compile Pareto sweep results.')
    parser.add_argument('results_dir', help='Directory containing one subfolder per run')
    parser.add_argument('--last-n', type=int, default=20,
                        help='Number of last episodes to average (default: 20)')
    parser.add_argument('--out', default=None,
                        help='Output CSV path (default: {MODEL}_sweep_data.csv '
                             'in results_dir)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} is not a directory")
        sys.exit(1)

    subfolders = sorted(f for f in results_dir.iterdir() if f.is_dir())
    if not subfolders:
        print(f"ERROR: no subfolders found in {results_dir}")
        sys.exit(1)

    # infer model name from folder names
    model_name = infer_model_name([f.name for f in subfolders])
    out_path = Path(args.out) if args.out else results_dir / f'{model_name}_sweep_data.csv'

    rows = []
    skipped = []

    for folder in subfolders:
        weights = extract_weights(folder.name)
        if weights is None:
            print(f"[SKIP] {folder.name} — could not extract wb/wp weights from name")
            skipped.append(folder.name)
            continue

        w_bike, w_ped = weights
        metrics_file = find_metrics_file(folder)
        if metrics_file is None:
            print(f"[SKIP] {folder.name} — no *_mm_step_metrics* file found")
            skipped.append(folder.name)
            continue

        print(f"[OK]   {folder.name}")
        print(f"       wb={w_bike}, wp={w_ped}, metrics={metrics_file.name}")

        result = process_run(folder, metrics_file, args.last_n)
        if result is None:
            skipped.append(folder.name)
            continue

        rows.append({
            'w_bike':    w_bike,
            'w_ped':     w_ped,
            'car_wait':  round(result['car_wait'],  4),
            'bike_wait': round(result['bike_wait'], 4),
            'ped_wait':  round(result['ped_wait'],  4),
        })

        print(f"       car={result['car_wait']:.2f}  "
              f"bike={result['bike_wait']:.2f}  "
              f"ped={result['ped_wait']:.2f}  "
              f"(avg last {args.last_n} eps)")

    if not rows:
        print("\nERROR: no valid runs found — nothing to write")
        sys.exit(1)

    out_df = pd.DataFrame(rows, columns=['w_bike', 'w_ped', 'car_wait', 'bike_wait', 'ped_wait'])
    out_df = out_df.sort_values(['w_bike', 'w_ped']).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Compiled {len(rows)} runs ({len(skipped)} skipped)")
    if skipped:
        print(f"Skipped: {skipped}")
    print(f"Output:  {out_path}")
    print(f"\nPreview:\n{out_df.to_string(index=False)}")
    print(f"\nRun pareto sweep with:")
    print(f"  python pareto_sweep.py {out_path}")


if __name__ == '__main__':
    main()