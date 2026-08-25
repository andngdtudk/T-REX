"""Incident configuration schema for TrexEnv (see trex_env.py).

Base (no-incident) and incident scenarios are two instances of the same schema:
an `IncidentConfig` (incidents enabled, with these parameters) or `None`
(incidents disabled entirely -- the Initializer/Deployment subsystem in
T_REX.py is never constructed or invoked, consuming zero RNG draws; see
trex_env.py::TrexEnv for why that distinction matters for reproducibility).
"""
from dataclasses import dataclass


@dataclass
class IncidentConfig:
    """Configuration for TrexEnv's incident subsystem.

    Attributes
    ----------
    level : int
        Number of simultaneous incidents to sample per episode (passed to
        T_REX.py::Initializer as `level`, and to `Deployment` indirectly).
        Section 2.3 of the paper describes single-incident episodes; this
        repo's pre-existing default of 2 predates this audit and is kept
        unchanged here, not altered by this refactor.
    is_random : bool
        Whether each incident's edge/lanes/position/time/duration are
        randomly sampled (True, the normal training/eval mode) or must be
        supplied via a fixed pre_seed (False is not currently exercised by
        main.py, kept for parity with the pre-refactor default).
    """
    level: int = 2
    is_random: bool = True


# Example configs demonstrating both modes of the shared schema (point 4 of the
# unified-environment design brief). Pass NO_INCIDENTS for the base scenario,
# DEFAULT_INCIDENTS for the incident scenario -- main.py does exactly this,
# keyed off --strategy.
NO_INCIDENTS = None
DEFAULT_INCIDENTS = IncidentConfig(level=2)
