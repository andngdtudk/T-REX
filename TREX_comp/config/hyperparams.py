"""Per-network learning-rate defaults matching Appendix B of the T-REX paper.

Only IDQN and MPLight are covered here: these are the two methods for which the
audit task brief supplied an explicit per-network learning-rate grid. FMA2C/IPPO
hyperparameters were not part of that brief and are left to their own class/config
defaults (see TREX_comp/config/agent_config.py and TREX_comp/agents/pfrl_ppo.py).

'Ingolstadt Region' in the paper corresponds to the 21-intersection 'ingolstadt21'
network (vs. the 7-intersection 'ingolstadt7' Corridor); 'Grid4x4' is 'grid4x4'.
All other networks fall under the paper's 'elsewhere'/'others' bucket.
"""

LEARNING_RATE_DEFAULTS = {
    'IDQN': {
        'grid4x4': 1e-5,
        'ingolstadt21': 1e-5,
        '__default__': 0.001,
    },
    'MPLight': {
        'grid4x4': 0.005,
        'ingolstadt21': 0.01,
        '__default__': 0.001,
    },
}


def resolve_learning_rate(agent_name, map_name, override=None):
    """Resolve the learning rate to use for a given agent/network combination.

    Parameters
    ----------
    agent_name : str
        Name of the agent class (e.g. 'IDQN', 'MPLight').
    map_name : str
        Name of the network/map (e.g. 'grid4x4', 'ingolstadt21').
    override : float or None
        An explicit value (e.g. from a CLI flag) that always takes precedence
        when not None.

    Returns
    -------
    float or None
        The learning rate to use, or None if `agent_name` has no entry here
        (in which case the caller should fall back to the agent class's own
        default rather than pass an explicit value).
    """
    if override is not None:
        return override

    agent_defaults = LEARNING_RATE_DEFAULTS.get(agent_name)
    if agent_defaults is None:
        return None

    return agent_defaults.get(map_name, agent_defaults['__default__'])
