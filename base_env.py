"""Deprecated. BaseEnv has been folded into trex_env.py::TrexEnv.

This module is kept only for backward compatibility with code importing
`from base_env import BaseEnv` directly; it will be removed in a future
release. New code should use:

    from trex_env import TrexEnv
    env = TrexEnv(..., incident_config=None)  # incidents disabled

See trex_env.py and AUDIT_REPORT.md for why this merge happened and what,
if anything, changed in the process (short version: one genuine bug --
BaseEnv's route-file path construction -- was fixed, not preserved; see
TrexEnv's class docstring for the details).
"""
import warnings

from trex_env import TrexEnv


class BaseEnv(TrexEnv):
    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False, end_time=3600,
                 step_length=10, yellow_length=4, step_ratio=1, max_distance=200, lights=(), log_dir='/',
                 libsumo=False, warmup=0, gymma=False, run=0, level=None, sumo_seed=None):
        warnings.warn(
            "base_env.BaseEnv is deprecated; use trex_env.TrexEnv(..., incident_config=None) instead. "
            "This wrapper will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # `level` was accepted by the pre-refactor BaseEnv but never used (incidents
        # were simply never wired up) -- dropped here rather than forwarded.
        super().__init__(
            run_name=run_name, map_name=map_name, net=net, state_fn=state_fn, reward_fn=reward_fn,
            route=route, gui=gui, end_time=end_time, step_length=step_length, yellow_length=yellow_length,
            step_ratio=step_ratio, max_distance=max_distance, lights=lights, log_dir=log_dir,
            libsumo=libsumo, warmup=warmup, gymma=gymma, run=run,
            incident_config=None, sumo_seed=sumo_seed,
        )
