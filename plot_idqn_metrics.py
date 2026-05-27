import argparse
import os
import re
import matplotlib.pyplot as plt

# usage:
# python main.py --agent IDQN_DELTA --map kbh_joined_432 --eps 1 --tr 0 --strategy 1 --max_green_hold 12 | Tee-Object -FilePath logs\idqn_delta.log 
# 
"""
python plot_idqn_metrics.py `
>>   --log logs\idqn_deltasclip224.log ` 
>>   --log logs\idqn_deltasclip55.log `
>>   --label IDQN_DELTASClip224 `
>>   --label IDQN_DELTASCLIP55 `
>>   --plots_dir plots `
>>   --prefix idqn_compare
 """
STEP_PATTERN = re.compile(r"step\s+(\d+):")
KV_PATTERN = re.compile(r"([a-zA-Z_]+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(log_path):
    step_metrics = {}

    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            with open(log_path, "r", encoding=encoding) as handle:
                for line in handle:
                    step_match = STEP_PATTERN.search(line)
                    if not step_match:
                        continue

                    step = int(step_match.group(1))
                    for key, value in KV_PATTERN.findall(line):
                        try:
                            metric_value = float(value)
                        except ValueError:
                            continue
                        step_metrics.setdefault(step, {})[key] = metric_value
            return step_metrics
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode log file")


def build_series(step_metrics, keys):
    steps = sorted(step_metrics.keys())
    series = {}
    for key in keys:
        series[key] = [step_metrics[step].get(key) for step in steps]
    return steps, series


def _resolve_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace(" ", "_")


def plot_metric_across_agents(agent_series, title, output_path, ylabel="Value"):
    plotted = False
    plt.figure(figsize=(10, 6))
    for label, steps, values in agent_series:
        if not steps:
            continue
        plotted = True
        plt.plot(steps, values, linewidth=1.6, label=label)

    if not plotted:
        plt.close()
        return False

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_q_band_across_agents(agent_series, title, output_path):
    plotted = False
    plt.figure(figsize=(10, 6))
    for label, steps, q_min, q_max, q_mean in agent_series:
        if not steps:
            continue
        plotted = True
        plt.fill_between(steps, q_min, q_max, alpha=0.18)
        plt.plot(steps, q_mean, linewidth=1.8, label=label)

    if not plotted:
        plt.close()
        return False

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot IDQN training metrics from console logs.")
    parser.add_argument("--log", action="append", required=True, help="Path to a training log file.")
    parser.add_argument("--label", action="append", help="Optional label for each log (same order).")
    parser.add_argument("--plots_dir", default="plots", help="Output directory for plots.")
    parser.add_argument("--prefix", default="idqn", help="Filename prefix for saved plots.")
    args = parser.parse_args()

    labels = args.label or []
    if labels and len(labels) != len(args.log):
        raise ValueError("Provide the same number of --label entries as --log entries.")

    logs = []
    for idx, log_path in enumerate(args.log):
        step_metrics = parse_log(log_path)
        if not step_metrics:
            print(f"No step metrics found in {log_path}.")
            continue
        label = labels[idx] if labels else _resolve_label(log_path)
        logs.append((label, step_metrics))

    if not logs:
        return

    os.makedirs(args.plots_dir, exist_ok=True)

    metric_aliases = {
        "entropy": ["action_entropy", "entropy"],
        "loss": ["average_loss", "loss"],
        "td_error": ["td_error"],
        "avg_q": ["average_q", "avg_q", "q"],
        "total_wait": ["total_wait"],
        "total_queue": ["total_queue"],
        "total_reward": ["total_reward"],
    }

    q_series = []
    for label, step_metrics in logs:
        steps, series = build_series(step_metrics, ["q_min", "q_max", "q_mean"])
        qs = list(zip(steps, series.get("q_min", []), series.get("q_max", []), series.get("q_mean", [])))
        xs = []
        mins = []
        maxs = []
        means = []
        for step, v_min, v_max, v_mean in qs:
            if v_min is None or v_max is None or v_mean is None:
                continue
            xs.append(step)
            mins.append(v_min)
            maxs.append(v_max)
            means.append(v_mean)
        q_series.append((label, xs, mins, maxs, means))

    q_output = os.path.join(args.plots_dir, f"{args.prefix}_q_values.png")
    plot_q_band_across_agents(q_series, "q value range", q_output)

    for metric_name, aliases in metric_aliases.items():
        agent_series = []
        for label, step_metrics in logs:
            steps = sorted(step_metrics.keys())
            values = []
            for step in steps:
                value = None
                for key in aliases:
                    if key in step_metrics[step]:
                        value = step_metrics[step][key]
                        break
                values.append(value)

            xs = [step for step, value in zip(steps, values) if value is not None]
            ys = [value for value in values if value is not None]
            agent_series.append((label, xs, ys))

        output_path = os.path.join(args.plots_dir, f"{args.prefix}_{metric_name}.png")
        plot_metric_across_agents(agent_series, metric_name, output_path)

    print(f"Plots saved to {args.plots_dir}.")


if __name__ == "__main__":
    main()
