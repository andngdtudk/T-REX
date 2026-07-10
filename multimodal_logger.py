"""
multimodal_logger.py
--------------------
Episode- and step-level metrics logger for multimodal traffic-signal agents.

Captures per-mode wait/queue breakdowns (car, bike, pedestrian), reward
signal composition, state-feature statistics, and agent internals so you
can diagnose whether bikes/peds are swamping the reward or state.

Works for ANY agent, not just IDQN_MM2 -- agent internals (epsilon, mean_q,
loss) are extracted best-effort and simply come back empty/NaN for agents
that don't expose them. Files are namespaced by `agent_name` so multiple
agents can log into the same `log_dir` without clobbering each other.

Usage
-----
Drop this file next to main.py, then in main.py replace the run_episode
call inside run_base_scenario / run_incident_scenario with:

    from multimodal_logger import MultimodalLogger

    mm_logger = MultimodalLogger(log_dir=agt_config['log_dir'], agent_name=args.agent)

    # inside the episode loop:
    stats = run_episode(env, agent, episode_index=ep + 1,
                        log_state_csv=log_state_csv,
                        mm_logger=mm_logger,       # <-- add this
                        episode_num=ep + 1)        # <-- add this
    mm_logger.end_episode(ep + 1)

And update run_episode's signature (see patch at bottom of this file).
"""

import csv
import os
import re
import time
from collections import defaultdict
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultimodalLogger:
    """Accumulates per-step metrics and flushes episode summaries to CSV.

    Instantiate once per agent/run. `agent_name` is sanitized and used as a
    filename prefix (e.g. "IDQN_MM2_mm_episode_metrics.csv") so several
    agents can safely share the same `log_dir`.
    """

    # ------------------------------------------------------------------ #
    # Construction / teardown                                              #
    # ------------------------------------------------------------------ #

    def __init__(self, log_dir: str, agent_name: str = "agent"):
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._agent_name = agent_name
        self._prefix = _sanitize_filename(agent_name)

        # -- open CSV files once ----------------------------------------
        self._episode_file, self._episode_writer = self._open_csv(
            f"{self._prefix}_mm_episode_metrics.csv",
            [
                "episode", "agent",
                # absolute waits (sum over all signals x lanes, in seconds)
                "car_wait_total", "bike_wait_total", "ped_wait_total",
                # queues
                "car_queue_total", "bike_queue_total", "ped_queue_total",
                # normalised reward components (mean over signals)
                "reward_mean", "reward_std", "reward_min", "reward_max",
                # mode fractions  (mode_wait / total_wait, mode_queue / total_queue)
                "bike_wait_frac", "ped_wait_frac",
                "bike_queue_frac", "ped_queue_frac",
                # episode-level agent internals (mean over steps)
                "epsilon", "mean_q", "mean_loss",
                # timing
                "decisions", "wall_seconds",
            ],
        )

        self._step_file, self._step_writer = self._open_csv(
            f"{self._prefix}_mm_step_metrics.csv",
            [
                "episode", "agent", "decision",
                "car_wait", "bike_wait", "ped_wait",
                "car_queue", "bike_queue", "ped_queue",
                "reward_mean", "reward_min", "reward_max",
                "epsilon", "mean_q",
            ],
        )

        self._reward_dist_file, self._reward_dist_writer = self._open_csv(
            f"{self._prefix}_mm_reward_dist.csv",
            ["episode", "agent", "decision", "signal_id", "reward"],
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
        agent=None,            # optional: any agent instance
    ):
        """Record one decision step."""

        # --- raw mode waits & queues from environment signals ----------
        car_wait = bike_wait = ped_wait = 0.0
        car_queue = bike_queue = ped_queue = 0.0

        for signal in env.signals.values():
            ped_wait += float(signal.full_observation.get("ped_total_wait", 0.0))

            # Pedestrian "queue" equivalent: system-wide sum of per-phase
            # pedestrian crossing pressure (traci getServedPersonCount),
            # computed in Signal.observe(). Prefer "ped_crossing_pressure_waiting"
            # (currently-served phase zeroed out, so people actively crossing
            # right now aren't counted as "waiting") over the raw
            # "ped_crossing_pressure", falling back to the raw field for
            # Signal implementations that predate the _waiting variant.
            #
            # CAVEAT (still applies even with the waiting variant): if two
            # different phase-pairs serve overlapping pedestrian groups, the
            # same waiting people can be counted under both pair_idx entries,
            # so this sum can overcount true distinct waiting pedestrians --
            # unlike car_queue/bike_queue, where each vehicle is attributed to
            # exactly one lane. Treat ped_queue as a pressure-based proxy, not
            # an exact headcount.
            ped_pressure = signal.full_observation.get("ped_crossing_pressure_waiting")
            if ped_pressure is None:
                ped_pressure = signal.full_observation.get("ped_crossing_pressure", {})
            if isinstance(ped_pressure, dict):
                ped_queue += float(sum(ped_pressure.values()))

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

        # --- agent internals (best-effort; NaN/0 if agent doesn't expose) --
        epsilon = _extract_epsilon(agent)
        mean_q  = _extract_mean_q(agent)

        # -- accumulate for episode summary -----------------------------
        self._acc["car_wait"]   += car_wait
        self._acc["bike_wait"]  += bike_wait
        self._acc["ped_wait"]   += ped_wait
        self._acc["car_queue"]  += car_queue
        self._acc["bike_queue"] += bike_queue
        self._acc["ped_queue"]  += ped_queue
        self._acc["reward_sum"] += rew_mean
        self._acc["reward_sq"]  += rew_mean ** 2
        self._acc["reward_min"]  = min(self._acc["reward_min"], rew_min)
        self._acc["reward_max"]  = max(self._acc["reward_max"], rew_max)
        self._acc["epsilon"]    += epsilon if not np.isnan(epsilon) else 0.0
        self._acc["mean_q"]     += mean_q if not np.isnan(mean_q) else 0.0
        self._acc["loss"]       += _extract_loss(agent)
        self._acc["steps"]      += 1

        # -- flush to step CSV ------------------------------------------
        self._step_writer.writerow({
            "episode": episode, "agent": self._agent_name, "decision": decision,
            "car_wait": round(car_wait, 3),
            "bike_wait": round(bike_wait, 3),
            "ped_wait": round(ped_wait, 3),
            "car_queue": round(car_queue, 3),
            "bike_queue": round(bike_queue, 3),
            "ped_queue": round(ped_queue, 3),
            "reward_mean": round(rew_mean, 5),
            "reward_min": round(rew_min, 5),
            "reward_max": round(rew_max, 5),
            "epsilon": round(epsilon, 4) if not np.isnan(epsilon) else "",
            "mean_q": round(mean_q, 5) if not np.isnan(mean_q) else "",
        })

        # -- per-signal reward distribution (sampled every 50 steps) ---
        if decision % 50 == 0 and isinstance(rewards, dict):
            for sig_id, rew in rewards.items():
                self._reward_dist_writer.writerow({
                    "episode": episode, "agent": self._agent_name, "decision": decision,
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
        total_queue = (
            self._acc["car_queue"] + self._acc["bike_queue"] + self._acc["ped_queue"]
        ) or 1.0  # avoid div-by-zero

        self._episode_writer.writerow({
            "episode":        episode,
            "agent":          self._agent_name,
            "car_wait_total": round(self._acc["car_wait"],  2),
            "bike_wait_total":round(self._acc["bike_wait"], 2),
            "ped_wait_total": round(self._acc["ped_wait"],  2),
            "car_queue_total":round(self._acc["car_queue"], 2),
            "bike_queue_total":round(self._acc["bike_queue"],2),
            "ped_queue_total":round(self._acc["ped_queue"], 2),
            "reward_mean":    round(rsum / n, 5),
            "reward_std":     round(np.sqrt(max(rsq/n - (rsum/n)**2, 0.0)), 5),
            "reward_min":     round(self._acc["reward_min"], 5),
            "reward_max":     round(self._acc["reward_max"], 5),
            "bike_wait_frac": round(self._acc["bike_wait"] / total_wait, 4),
            "ped_wait_frac":  round(self._acc["ped_wait"]  / total_wait, 4),
            "bike_queue_frac":round(self._acc["bike_queue"] / total_queue, 4),
            "ped_queue_frac": round(self._acc["ped_queue"]  / total_queue, 4),
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

def _sanitize_filename(name: str) -> str:
    """Make an agent name safe to use as a filename prefix."""
    name = str(name).strip() or "agent"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _extract_mean_q(agent) -> float:
    """Best-effort extraction of mean Q-value from any agent."""
    if agent is None:
        return float("nan")
    # training_stats() path used elsewhere in main.py (IDQN, MPLight, ...)
    if hasattr(agent, "training_stats"):
        try:
            stats = agent.training_stats() or {}
        except Exception:
            stats = {}
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

def _extract_epsilon(agent) -> float:
    """Read epsilon from any agent that exposes it (training_stats or attr)."""
    if agent is None:
        return float("nan")
    if hasattr(agent, "training_stats"):
        try:
            stats = agent.training_stats() or {}
        except Exception:
            stats = {}
        if "epsilon" in stats:
            return float(stats["epsilon"])
    # direct attribute fallback
    for attr in ("epsilon", "_epsilon"):
        if hasattr(agent, attr):
            v = getattr(agent, attr)
            if v is not None:
                return float(v)
    return float("nan")


def _extract_loss(agent) -> float:
    if agent is None:
        return 0.0
    if hasattr(agent, "training_stats"):
        try:
            stats = agent.training_stats() or {}
        except Exception:
            stats = {}
        for key in ("average_loss", "loss"):
            if key in stats:
                return float(stats[key])
    return 0.0


# ---------------------------------------------------------------------------
# Patch for run_episode in main.py
# ---------------------------------------------------------------------------
# Replace the existing run_episode function with this version.
# The only additions are the `mm_logger` and `episode_num` parameters and
# the block feeding the MultimodalLogger.

def run_episode(env, agent, obs=None, episode_index=None, log_state_csv=None,
                mm_logger=None, episode_num=None):
    """Drop-in replacement for run_episode that feeds the MultimodalLogger."""
    import os as _os
    debug_episode = _os.getenv("TREX_DEBUG_EPISODE", "").strip().lower() in {"1", "true", "yes", "on"}
    try:
        from main import _log_state_features as _main_log_state_features
        from main import _log_step_reward as _main_log_step_reward
        from main import _log_step_metrics as _main_log_step_metrics
    except Exception:
        _main_log_state_features = None
        _main_log_step_reward = None
        _main_log_step_metrics = None

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

        # ---- multimodal logging (works for any agent) ------------------
        if mm_logger is not None:
            mm_logger.log_step(
                episode=episode_num or (episode_index or 0),
                decision=decisions + 1,
                env=env,
                rewards=rew,
                agent=agent,
            )
        # ---------------------------------------------------------------

        # keep existing helpers when the refactored main module exposes them
        if _main_log_state_features is not None:
            _main_log_state_features(obs, episode_index, decisions + 1, log_state_csv)
        if _main_log_step_reward is not None:
            try:
                _main_log_step_reward(env, rew, decisions + 1)
            except Exception:
                pass
        agent.observe(obs, rew, done, info)
        if _main_log_step_metrics is not None:
            _main_log_step_metrics(agent, step_metrics, decisions + 1)
        decisions += 1

        if decisions % 100 == 0:
            print(
                f"  progress: decisions={decisions}, "
                f"sim_time={env.sumo.simulation.getTime():.1f}",
                flush=True,
            )

    sim_time = env.sumo.simulation.getTime() if hasattr(env, "sumo") else -1
    return {"decisions": decisions, "sim_time": sim_time, "done": done}


# Backward-compatible alias for older refactors that imported the helper
# under the previous name.
run_episode_mm = run_episode


__all__ = ["MultimodalLogger", "run_episode", "run_episode_mm"]