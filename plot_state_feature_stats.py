import argparse
import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

"""
Plot diagnostics from state_feature_stats.csv files.
Usage win:
python plot_state_feature_stats.py --glob "results/**/state_feature_stats.csv" --plots_dir plots/state_diagnostics

Usage linux:
python plot_state_feature_stats.py \
    --glob "results/**/state_feature_stats.csv" \
    --plots_dir plots/state_diagnostics

Optional arguments:
--feature phase --feature total_wait to limit features
--max_decision 300 to cap decisions
--window 25 to adjust smoothing
--wandb to log to Weights & Biases (wandb)

phase: binary flag per lane; 1 if the lane index equals the current signal phase index, else 0.
approach: count of vehicles on the lane that are moving (not queued) within detector range.
total_wait: sum of waiting times (seconds) of all vehicles currently queued on the lane.
queue: count of queued vehicles on the lane (vehicles with wait > 0).
total_speed: sum of speeds (m/s) of all vehicles detected on the lane.
"""


def _sanitize_filename(value):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _rolling(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def load_csvs(pattern):
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        return pd.DataFrame(), []

    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        run_name = Path(path).parent.name
        df["run"] = run_name
        df["source"] = path
        frames.append(df)

    if not frames:
        return pd.DataFrame(), []

    return pd.concat(frames, ignore_index=True), paths


def plot_feature_series(grouped, feature, metric, plots_dir, window):
    plt.figure(figsize=(11, 6))
    plotted = False
    for run_name, frame in grouped:
        data = frame[frame["feature"] == feature]
        if data.empty:
            continue
        data = data.sort_values("decision")
        series = _rolling(data[metric], window)
        plt.plot(data["decision"], series, linewidth=1.6, label=run_name)
        plotted = True

    if not plotted:
        plt.close()
        return False, None

    plt.xlabel("Decision")
    plt.ylabel(metric)
    plt.title(f"{feature} - {metric}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    filename = f"feature_{_sanitize_filename(feature)}_{metric}.png"
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True, output_path


def plot_feature_band(grouped, feature, plots_dir, window):
    plt.figure(figsize=(11, 6))
    plotted = False
    for run_name, frame in grouped:
        data = frame[frame["feature"] == feature]
        if data.empty:
            continue
        data = data.sort_values("decision")
        mean = _rolling(data["mean"], window)
        p05 = _rolling(data["p05"], window)
        p95 = _rolling(data["p95"], window)
        plt.fill_between(data["decision"], p05, p95, alpha=0.18)
        plt.plot(data["decision"], mean, linewidth=1.6, label=run_name)
        plotted = True

    if not plotted:
        plt.close()
        return False, None

    plt.xlabel("Decision")
    plt.ylabel("value")
    plt.title(f"{feature} - mean with p05/p95 band")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    filename = f"feature_{_sanitize_filename(feature)}_band.png"
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True, output_path


def _parse_tags(value):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def log_wandb(grouped, run_name, metrics, args):
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is not available; install it with `pip install wandb`.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        group=args.wandb_group,
        tags=_parse_tags(args.wandb_tags),
        mode=args.wandb_mode,
        config={
            "glob": args.glob,
            "window": args.window,
            "max_decision": args.max_decision,
            "features": args.feature,
        },
        reinit=True,
    )

    run_frame = grouped[grouped["run"] == run_name]
    for decision, frame in run_frame.groupby("decision"):
        payload = {"decision": int(decision)}
        for _, row in frame.iterrows():
            feature = row["feature"]
            for metric in metrics:
                payload[f"{feature}/{metric}"] = float(row[metric])
        wandb.log(payload, step=int(decision))

    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics from state_feature_stats.csv files."
    )
    parser.add_argument(
        "--glob",
        default="results/**/state_feature_stats.csv",
        help="Glob pattern for CSV files.",
    )
    parser.add_argument(
        "--plots_dir",
        default="plots/state_diagnostics",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling mean window for smoothing.",
    )
    parser.add_argument(
        "--max_decision",
        type=int,
        default=None,
        help="Optional max decision to include.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        help="Feature name to include (repeatable).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log per-decision metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project",
        default="trex-state-features",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="Weights & Biases entity/team (optional).",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="Weights & Biases run name (defaults to run folder).",
    )
    parser.add_argument(
        "--wandb_group",
        default=None,
        help="Weights & Biases group name (optional).",
    )
    parser.add_argument(
        "--wandb_tags",
        default=None,
        help="Comma-separated list of Weights & Biases tags.",
    )
    parser.add_argument(
        "--wandb_mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    args = parser.parse_args()

    df, paths = load_csvs(args.glob)
    if df.empty:
        print(f"No CSVs found for pattern: {args.glob}")
        return

    if args.max_decision is not None:
        df = df[df["decision"] <= args.max_decision]

    if args.feature:
        df = df[df["feature"].isin(args.feature)]

    if df.empty:
        print("No data left after filtering.")
        return

    os.makedirs(args.plots_dir, exist_ok=True)

    numeric_cols = [
        "mean", "std", "min", "max", "p05", "p95",
        "nan_count", "inf_count", "zero_frac"
    ]

    grouped = (
        df.groupby(["run", "feature", "decision"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["run", "feature", "decision"])
    )

    features = sorted(grouped["feature"].unique())
    run_groups = [(name, frame) for name, frame in grouped.groupby("run")]

    metrics = ["mean", "std", "min", "max", "zero_frac", "nan_count", "inf_count"]

    plotted_any = False
    for feature in features:
        for metric in metrics:
            plotted, output_path = plot_feature_series(
                run_groups, feature, metric, args.plots_dir, args.window
            )
            plotted_any |= plotted
        plotted, output_path = plot_feature_band(run_groups, feature, args.plots_dir, args.window)
        plotted_any |= plotted

    if args.wandb:
        for run_name, _ in run_groups:
            log_wandb(grouped, run_name, metrics, args)

    if plotted_any:
        print(f"Plots saved to {args.plots_dir}")
    else:
        print("No plots generated. Check filters and data.")


if __name__ == "__main__":
    main()
