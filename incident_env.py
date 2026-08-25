"""Deprecated. IncidentEnv has been folded into trex_env.py::TrexEnv.

This module is kept only for backward compatibility with code importing
`from incident_env import IncidentEnv` directly; it will be removed in a
future release. New code should use:

    from trex_env import TrexEnv
    from TREX_comp.config.incident_config import IncidentConfig
    env = TrexEnv(..., incident_config=IncidentConfig(level=2))  # incidents enabled

See trex_env.py and AUDIT_REPORT.md for why this merge happened.
"""
import warnings

from trex_env import TrexEnv
from TREX_comp.config.incident_config import IncidentConfig


class IncidentEnv(TrexEnv):
    """Traffic signal control environment with incident simulation based on SUMO."""

    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False,
                 end_time=3600, step_length=10, yellow_length=4, step_ratio=1,
                 max_distance=300, lights=(), log_dir='/', libsumo=False, warmup=100, gymma=False, run=0, level=2,
                 sumo_seed=None):
        warnings.warn(
            "incident_env.IncidentEnv is deprecated; use "
            "trex_env.TrexEnv(..., incident_config=IncidentConfig(level=...)) instead. "
            "This wrapper will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            run_name=run_name, map_name=map_name, net=net, state_fn=state_fn, reward_fn=reward_fn,
            route=route, gui=gui, end_time=end_time, step_length=step_length, yellow_length=yellow_length,
            step_ratio=step_ratio, max_distance=max_distance, lights=lights, log_dir=log_dir,
            libsumo=libsumo, warmup=warmup, gymma=gymma, run=run,
            incident_config=IncidentConfig(level=level), sumo_seed=sumo_seed,
        )
