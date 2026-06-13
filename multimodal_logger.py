"""
multimodal_logger.py
--------------------
Episode- and step-level metrics logger for the IDQN_MM2 multimodal agent.

Captures per-mode wait/queue breakdowns, reward signal composition,
state-feature statistics, and agent internals so you can diagnose
whether bikes/peds are swamping the reward or state.

Usage
-----
Drop this file next to main.py, then in main.py replace the run_episode
call inside run_base_scenario / run_incident_scenario with:

    from multimodal_logger import MultimodalLogger

    logger = MultimodalLogger(log_dir=agt_config['log_dir'], agent_name='IDQN_MM2')

    # inside the episode loop:
    stats = run_episode(env, agent, episode_index=ep + 1,
                        log_state_csv=log_state_csv,
                        mm_logger=logger,          # <-- add this
                        episode_num=ep + 1)        # <-- add this
    logger.end_episode(ep + 1)

And update run_episode's signature (see patch at bottom of this file).
"""

import csv
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultimodalLogger:
    """Accumulates per-step metrics and flushes episode summaries to CSV."""

    # ------------------------------------------------------------------ #
    # Construction / teardown                                              #
    # ------------------------------------------------------------------ #

    def __init__(self, log_dir: str, agent_name: str = "IDQN_MM2"):
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._agent_name = agent_name

        # -- open CSV files once ----------------------------------------
        self._episode_file, self._episode_writer = self._open_csv(
            "mm_episode_metrics.csv",
            [
                "episode",
                # absolute waits (sum over all signals × lanes, in seconds)
                "car_wait_total", "bike_wait_total", "ped_wait_total",
                # queues
                "car_queue_total", "bike_queue_total",
                # normalised reward components (mean over signals)
                "reward_mean", "reward_std", "reward_min", "reward_max",
                # mode fractions  (bike_wait / (car_wait + bike_wait + ped_wait))
                "bike_wait_frac", "ped_wait_frac",
                # episode-level agent internals (mean over steps)
                "epsilon", "mean_q", "mean_loss",
                # timing
                "decisions", "wall_seconds",
            ],
        )

        self._step_file, self._step_writer = self._open_csv(
            "mm_step_metrics.csv",
            [
                "episode", "decision",
                "car_wait", "bike_wait", "ped_wait",
                "car_queue", "bike_queue",
                "reward_mean", "reward_min", "reward_max",
                "epsilon", "mean_q",
            ],
        )

        self._reward_dist_file, self._reward_dist_writer = self._open_csv(
            "mm_reward_dist.csv",
            ["episode", "decision", "signal_id", "reward"],
        )

        # -- per-episode accumulator ------------------------------------
        self._reset_accumulators()
        self._ep_start_time: float = time.time()

    def close(self):
        for fh in (self._episode_file, self._step_file, self._reward_dist_file):
            fh.close()

    # ------------------------------------------------------------------ #
    # Called once per decision step from inside run_episode              #
    # ------------------------------------------------------------------ #

    def log_step(
        self,
        episode: int,
        decision: int,
        env,                   # BaseEnv / IncidentEnv instance
        rewards: dict,         # signal_id -> scalar reward
        agent=None,            # optional: IDQN instance
    ):
        """Record one decision step."""

        # --- raw mode waits & queues from environment signals ----------
        car_wait = bike_wait = ped_wait = 0.0
        car_queue = bike_queue = 0.0

        for signal in env.signals.values():
            ped_wait += float(signal.full_observation.get("ped_total_wait", 0.0))
            for lane in signal.lanes:
                obs = signal.full_observation[lane]
                tw   = float(obs.get("total_wait",      0.0))
                bw   = float(obs.get("bike_total_wait", 0.0))
                car_wait   += max(0.0, tw - bw)
                bike_wait  += bw
                car_queue  += float(obs.get("queue",      0.0))
                bike_queue += float(obs.get("bike_queue", 0.0))

        # --- reward stats across agents --------------------------------
        rew_values = list(rewards.values()) if isinstance(rewards, dict) else []
        rew_mean = float(np.mean(rew_values)) if rew_values else 0.0
        rew_min  = float(np.min(rew_values))  if rew_values else 0.0
        rew_max  = float(np.max(rew_values))  if rew_values else 0.0

        # --- agent internals -------------------------------------------
        epsilon = float(getattr(agent, "epsilon", float("nan")))
        mean_q  = _extract_mean_q(agent)

        # -- accumulate for episode summary -----------------------------
        self._acc["car_wait"]   += car_wait
        self._acc["bike_wait"]  += bike_wait
        self._acc["ped_wait"]   += ped_wait
        self._acc["car_queue"]  += car_queue
        self._acc["bike_queue"] += bike_queue
        self._acc["reward_sum"] += rew_mean
        self._acc["reward_sq"]  += rew_mean ** 2
        self._acc["reward_min"]  = min(self._acc["reward_min"], rew_min)
        self._acc["reward_max"]  = max(self._acc["reward_max"], rew_max)
        self._acc["epsilon"]    += epsilon
        self._acc["mean_q"]     += mean_q if not np.isnan(mean_q) else 0.0
        self._acc["loss"]       += _extract_loss(agent)
        self._acc["steps"]      += 1

        # -- flush to step CSV ------------------------------------------
        self._step_writer.writerow({
            "episode": episode, "decision": decision,
            "car_wait": round(car_wait, 3),
            "bike_wait": round(bike_wait, 3),
            "ped_wait": round(ped_wait, 3),
            "car_queue": round(car_queue, 3),
            "bike_queue": round(bike_queue, 3),
            "reward_mean": round(rew_mean, 5),
            "reward_min": round(rew_min, 5),
            "reward_max": round(rew_max, 5),
            "epsilon": round(epsilon, 4),
            "mean_q": round(mean_q, 5) if not np.isnan(mean_q) else "",
        })

        # -- per-signal reward distribution (sampled every 50 steps) ---
        if decision % 50 == 0 and isinstance(rewards, dict):
            for sig_id, rew in rewards.items():
                self._reward_dist_writer.writerow({
                    "episode": episode, "decision": decision,
                    "signal_id": sig_id, "reward": round(float(rew), 5),
                })

        # flush periodically so the file is readable during long runs
        if decision % 200 == 0:
            self._step_file.flush()
            self._reward_dist_file.flush()

    # ------------------------------------------------------------------ #
    # Called once at the end of each episode                             #
    # ------------------------------------------------------------------ #

    def end_episode(self, episode: int, decisions: int = 0, agent=None):
        n = max(self._acc["steps"], 1)
        rsum = self._acc["reward_sum"]
        rsq  = self._acc["reward_sq"]

        total_wait = (
            self._acc["car_wait"] + self._acc["bike_wait"] + self._acc["ped_wait"]
        ) or 1.0  # avoid div-by-zero

        self._episode_writer.writerow({
            "episode":        episode,
            "car_wait_total": round(self._acc["car_wait"],  2),
            "bike_wait_total":round(self._acc["bike_wait"], 2),
            "ped_wait_total": round(self._acc["ped_wait"],  2),
            "car_queue_total":round(self._acc["car_queue"], 2),
            "bike_queue_total":round(self._acc["bike_queue"],2),
            "reward_mean":    round(rsum / n, 5),
            "reward_std":     round(np.sqrt(max(rsq/n - (rsum/n)**2, 0.0)), 5),
            "reward_min":     round(self._acc["reward_min"], 5),
            "reward_max":     round(self._acc["reward_max"], 5),
            "bike_wait_frac": round(self._acc["bike_wait"] / total_wait, 4),
            "ped_wait_frac":  round(self._acc["ped_wait"]  / total_wait, 4),
            "epsilon":        round(self._acc["epsilon"] / n, 4),
            "mean_q":         round(self._acc["mean_q"]   / n, 5),
            "mean_loss":      round(self._acc["loss"]      / n, 6),
            "decisions":      decisions or n,
            "wall_seconds":   round(time.time() - self._ep_start_time, 1),
        })
        self._episode_file.flush()

        self._reset_accumulators()
        self._ep_start_time = time.time()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _open_csv(self, filename: str, fieldnames: list):
        path = os.path.join(self._log_dir, filename)
        fh = open(path, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        return fh, writer

    def _reset_accumulators(self):
        self._acc = defaultdict(float)
        self._acc["reward_min"] = float("inf")
        self._acc["reward_max"] = float("-inf")


# ---------------------------------------------------------------------------
# Agent introspection helpers (gracefully degrade if attrs not present)
# ---------------------------------------------------------------------------

def _extract_mean_q(agent) -> float:
    """Best-effort extraction of mean Q-value from IDQN agent."""
    if agent is None:
        return float("nan")
    # training_stats() path used elsewhere in main.py
    if hasattr(agent, "training_stats"):
        stats = agent.training_stats() or {}
        for key in ("average_q", "q", "mean_q"):
            if key in stats:
                return float(stats[key])
    # Direct attribute fallback
    for attr in ("mean_q", "avg_q", "_mean_q"):
        if hasattr(agent, attr):
            v = getattr(agent, attr)
            if v is not None:
                return float(v)
    return float("nan")


def _extract_loss(agent) -> float:
    if agent is None:
        return 0.0
    if hasattr(agent, "training_stats"):
        stats = agent.training_stats() or {}
        for key in ("average_loss", "loss"):
            if key in stats:
                return float(stats[key])
    return 0.0


# ---------------------------------------------------------------------------
# Patch for run_episode in main.py
# ---------------------------------------------------------------------------
# Replace the existing run_episode function with this version.
# The only additions are the `mm_logger` and `episode_num` parameters and
# two lines inside the loop body.

def run_episode_mm(env, agent, obs=None, episode_index=None, log_state_csv=None,
                   mm_logger=None, episode_num=None):
    """Drop-in replacement for run_episode that feeds the MultimodalLogger."""
    import os as _os
    debug_episode = _os.getenv("TREX_DEBUG_EPISODE", "").strip().lower() in {"1", "true", "yes", "on"}

    if obs is None:
        obs = env.reset()
    done = False
    decisions = 0

    while not done:
        step_metrics = agent.step_metrics(obs) if hasattr(agent, "step_metrics") else None
        act = agent.act(obs)

        safe_act = {}
        for signal_id in getattr(env, "signal_ids", []):
            selected = act.get(signal_id, 0)
            num_phases = len(env.phases.get(signal_id, []))
            safe_act[signal_id] = int(selected) % num_phases if num_phases > 0 else 0
        act = safe_act

        obs, rew, done, info = env.step(act)

        # ---- multimodal logging (new) ----------------------------------
        if mm_logger is not None:
            mm_logger.log_step(
                episode=episode_num or (episode_index or 0),
                decision=decisions + 1,
                env=env,
                rewards=rew,
                agent=agent,
            )
        # ---------------------------------------------------------------

        # keep existing helpers
        from main import _log_state_features, _log_step_reward, _log_step_metrics
        _log_state_features(obs, episode_index, decisions + 1, log_state_csv)
        _log_step_reward(env, rew, decisions + 1)
        agent.observe(obs, rew, done, info)
        _log_step_metrics(agent, step_metrics, decisions + 1)
        decisions += 1

        if decisions % 100 == 0:
            print(
                f"  progress: decisions={decisions}, "
                f"sim_time={env.sumo.simulation.getTime():.1f}",
                flush=True,
            )

    sim_time = env.sumo.simulation.getTime() if hasattr(env, "sumo") else -1
    return {"decisions": decisions, "sim_time": sim_time, "done": done}
