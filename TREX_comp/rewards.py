import numpy as np

from TREX_comp.config.mdp_config import mdp_configs

#region Waits

def wait(signals):
    """Local delay-minimization reward based on total waiting time.

    For each signal, this returns the negative sum of ``total_wait`` over its
    inbound lanes. It is unnormalized, so magnitude grows with demand/network
    size.

    Used by ``STOCHASTIC``, ``MAXWAVE``, and ``MAXPRESSURE``.
    """
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = -total_wait
    return rewards


def wait_norm(signals):
    """Normalized and clipped variant of :func:`wait`.

    Uses the same ``-total_wait`` objective, but scales by 224 and clips to
    ``[-4, 4]`` to stabilize optimization when reward magnitudes vary widely
    across episodes or maps.

    Used by ``IDQN`` and ``IPPO``.
    """
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = np.clip(-total_wait/224, -4, 4).astype(np.float32)
    return rewards

def _delta_wait(signals, prev_waits):
    """Compute raw delta waits and update previous-wait state."""
    rewards = dict()
    for signal_id in signals:
        total_wait = 0.0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        prev_wait = prev_waits.get(signal_id)
        if prev_wait is None:
            rewards[signal_id] = 0.0
        else:
            rewards[signal_id] = prev_wait - total_wait
        prev_waits[signal_id] = total_wait

    return rewards


def wait_delta(signals, prev_waits):
    """Delta wait reward: change in total_wait since last step.

    For each signal, this computes the change in total_wait across all inbound
    lanes since the last step. Used to avoid monotony as a result of stuck vehicles

    Used by ``IDQN_DELTA``.
    """
    rewards = _delta_wait(signals, prev_waits)
    for signal_id, value in rewards.items():
        rewards[signal_id] = np.float32(value)
    return rewards


def _get_delta_wait_config(config_key, require_clip=False):
    raw = mdp_configs.get(config_key)
    if not isinstance(raw, dict):
        raise ValueError(f"Missing mdp_configs['{config_key}'] for delta reward config")

    if 'norm_wait' not in raw:
        raise ValueError(f"Missing norm_wait in mdp_configs['{config_key}']")
    if require_clip and 'clip_wait' not in raw:
        raise ValueError(f"Missing clip_wait in mdp_configs['{config_key}']")

    cfg = {
        'norm_wait': raw['norm_wait'],
        'clip_wait': raw.get('clip_wait'),
    }
    return cfg

def wait_delta_norm(signals):
    """Stateful wrapper for :func:`wait_delta` with reset-safe bookkeeping.

    Keeps previous waits across steps and resets automatically if the set of
    signal IDs changes (for example, at episode boundaries).
    """
    prev_waits = getattr(wait_delta_norm, '_prev_waits', None)
    signal_ids = set(signals.keys())
    if prev_waits is None or set(prev_waits.keys()) != signal_ids:
        prev_waits = {}
        wait_delta_norm._prev_waits = prev_waits

    return wait_delta(signals, prev_waits)


def wait_delta_scale(signals):
    """Scaled delta wait reward for IDQN_DELTASCALE."""
    cfg = _get_delta_wait_config('IDQN_DELTASCALE')
    prev_waits = getattr(wait_delta_scale, '_prev_waits', None)
    signal_ids = set(signals.keys())
    if prev_waits is None or set(prev_waits.keys()) != signal_ids:
        prev_waits = {}
        wait_delta_scale._prev_waits = prev_waits

    rewards = _delta_wait(signals, prev_waits)
    for signal_id, value in rewards.items():
        rewards[signal_id] = np.float32(value / cfg['norm_wait'])
    return rewards


def wait_delta_sclip(signals):
    """Scaled + clipped delta wait reward for IDQN_DELTASCLIP."""
    cfg = _get_delta_wait_config('IDQN_DELTASCLIP', require_clip=True)
    prev_waits = getattr(wait_delta_sclip, '_prev_waits', None)
    signal_ids = set(signals.keys())
    if prev_waits is None or set(prev_waits.keys()) != signal_ids:
        prev_waits = {}
        wait_delta_sclip._prev_waits = prev_waits

    rewards = _delta_wait(signals, prev_waits)
    for signal_id, value in rewards.items():
        rewards[signal_id] = np.clip(
            value / cfg['norm_wait'],
            -cfg['clip_wait'],
            cfg['clip_wait'],
        ).astype(np.float32)
    return rewards


def _reset_wait_delta_norm():
    wait_delta_norm._prev_waits = {}


wait_delta_norm.reset = _reset_wait_delta_norm


def _reset_wait_delta_scale():
    wait_delta_scale._prev_waits = {}


def _reset_wait_delta_sclip():
    wait_delta_sclip._prev_waits = {}


wait_delta_scale.reset = _reset_wait_delta_scale
wait_delta_sclip.reset = _reset_wait_delta_sclip

#endregion
#============================================================================================
#region Wait multimodal

def _resolve_multimodal_wait_config():
    """Resolve config for multimodal waiting-time rewards.

    The function first checks ``mdp_configs['IDQN_MULTIMODAL']`` and falls back
    to ``mdp_configs['IDQN']`` if available. If neither provides scalar values,
    hard-coded defaults are used.
    """
    defaults = {
        'car_wait_weight': 1.0,
        'bike_wait_weight': 1.0,
        'ped_wait_weight': 1.0,
        'norm_wait': 224.0,
        'clip_wait': 4.0,
    }

    # Runtime map resolution in main.py typically overwrites mdp_configs[key]
    # with a scalar config dict; if that is not available, use defaults.
    for key in ('IDQN_MULTIMODAL', 'IDQN'):
        raw = mdp_configs.get(key)
        if not isinstance(raw, dict):
            continue

        if all(name in raw for name in ('car_wait_weight', 'bike_wait_weight', 'ped_wait_weight')):
            cfg = defaults.copy()
            cfg.update(raw)
            return cfg

    return defaults

def wait_multimodal_norm(signals):
    """Weighted, normalized wait reward over cars, bikes, and pedestrians.

    Per signal, this computes:
        combined_wait =
            car_wait_weight * car_wait
            + bike_wait_weight * bike_wait
            + ped_wait_weight * ped_wait

    where:
    - ``car_wait`` is lane ``total_wait - bike_total_wait`` (clipped at 0),
    - ``bike_wait`` is lane ``bike_total_wait``,
    - ``ped_wait`` is signal-level ``ped_total_wait``.

    Reward is ``-combined_wait / norm_wait`` clipped to
    ``[-clip_wait, clip_wait]``.

    Config source:
        ``mdp_configs['IDQN_MULTIMODAL']`` (preferred) or
        ``mdp_configs['IDQN']`` (fallback).
    """
    cfg = _resolve_multimodal_wait_config()
    rewards = dict()

    for signal_id, signal in signals.items():
        car_wait = 0.0
        bike_wait = 0.0
        for lane in signal.lanes:
            lane_wait = float(signal.full_observation[lane].get('total_wait', 0.0))
            lane_bike_wait = float(signal.full_observation[lane].get('bike_total_wait', 0.0))
            bike_wait += lane_bike_wait
            car_wait += max(0.0, lane_wait - lane_bike_wait)

        ped_wait = float(signal.full_observation.get('ped_total_wait', 0.0))

        combined_wait = (
            cfg['car_wait_weight'] * car_wait
            + cfg['bike_wait_weight'] * bike_wait
            + cfg['ped_wait_weight'] * ped_wait
        )

        rewards[signal_id] = np.clip(
            -combined_wait / cfg['norm_wait'],
            -cfg['clip_wait'],
            cfg['clip_wait'],
        ).astype(np.float32)

    return rewards

#endregion
#============================================================================================
#region Pressure and queue

def pressure(signals):
    """Traffic-pressure reward using upstream minus downstream queue.

    Computes queue pressure per signal as:
    inbound queue - reachable downstream outbound queue, then returns its
    negative. This encourages serving movements with larger local pressure
    imbalances rather than only minimizing absolute wait.

    Used by ``MPLight``, ``MPLightFULL``, and ``MPLightVAL``.
    """
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += signals[signal_id].full_observation[lane]['queue']

        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                queue_length -= signals[signal_id].signals[dwn_signal].full_observation[lane]['queue']

        rewards[signal_id] = -queue_length
    return rewards


def queue_maxwait(signals):
    """MA2C local reward combining queue length and max waiting penalty.

    Per signal reward is the negative weighted sum of lane queue and lane
     maximum waiting time:
     ``-(queue + coef * max_wait)``, where ``coef`` comes from
     ``mdp_configs['MA2C']['coef']``.

    MA2C-style worker reward component; not directly selected by any
     current ``--agent`` option in this repository.
    """
    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['MA2C']['coef'])
        rewards[signal_id] = -reward
    return rewards


def queue_maxwait_neighborhood(signals):
    """MA2C cooperative reward with downstream neighborhood shaping.

    Starts from :func:`queue_maxwait` and adds discounted rewards of immediate
     downstream neighbors using ``mdp_configs['MA2C']['coop_gamma']``.

    MA2C-style cooperative worker reward; not directly selected by any
     current ``--agent`` option in this repository.
    """
    rewards = queue_maxwait(signals)
    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                sum_reward += (mdp_configs['MA2C']['coop_gamma'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    return neighborhood_rewards

#endregion
#============================================================================================
#region FMAs

def _resolve_fma_config(config_key, signals):
    """Resolve and cache FMA-style hierarchical reward configuration.

    The function normalizes missing config fields (management groups,
    manager-neighbor relations, and worker-to-manager assignments) and writes
    the resolved result back into ``mdp_configs[config_key]`` so downstream
    reward calls can rely on a complete structure.

    Args:
        config_key: Key in ``mdp_configs`` (for example ``'FMA2C'``).
        signals: Mapping of signal_id to traffic-signal objects.

    Returns:
        A fully populated config dictionary with scalar coefficients and
        hierarchy mappings.
    """
    config = mdp_configs.get(config_key, {})

    # If config is not map-resolved yet or map has no hand-written entry,
    # fall back to a single manager that supervises all active signals.
    management = config.get('management') if isinstance(config, dict) else None
    if not management:
        management = {'top_mgr': list(signals.keys())}

    management_neighbors = config.get('management_neighbors') if isinstance(config, dict) else None
    if not management_neighbors:
        management_neighbors = {manager: [] for manager in management}
    else:
        manager_ids = set(management.keys())
        management_neighbors = {
            manager: [neighbor for neighbor in neighbors if neighbor in manager_ids]
            for manager, neighbors in management_neighbors.items()
        }
        for manager in management:
            management_neighbors.setdefault(manager, [])

    supervisors = config.get('supervisors') if isinstance(config, dict) else None
    if not supervisors:
        supervisors = {
            worker: manager
            for manager, workers in management.items()
            for worker in workers
        }

    default_manager = next(iter(management))
    for signal_id in signals:
        supervisors.setdefault(signal_id, default_manager)

    resolved = {
        'coef': config.get('coef', 0.4) if isinstance(config, dict) else 0.4,
        'coop_gamma': config.get('coop_gamma', 0.9) if isinstance(config, dict) else 0.9,
        'clip_wave': config.get('clip_wave', 4.0) if isinstance(config, dict) else 4.0,
        'clip_wait': config.get('clip_wait', 4.0) if isinstance(config, dict) else 4.0,
        'norm_wave': config.get('norm_wave', 5.0) if isinstance(config, dict) else 5.0,
        'norm_wait': config.get('norm_wait', 100.0) if isinstance(config, dict) else 100.0,
        'alpha': config.get('alpha', 0.75) if isinstance(config, dict) else 0.75,
        'management': management,
        'management_neighbors': management_neighbors,
        'supervisors': supervisors,
    }

    mdp_configs[config_key] = resolved
    return resolved

def fma2c(signals):
    """Hierarchical FMA2C reward for workers and managers.

    Produces a joint reward dictionary that includes:
    1) Worker (signal) rewards: local queue/max-wait penalties plus
       intra-region neighbor shaping via ``alpha``.
    2) Manager rewards: region-level terms based on fringe arrivals and
       liquidity (departures - arrivals), with inter-manager coupling.

    This variant reads coefficients from ``mdp_configs['FMA2C']``.

    Used by ``FMA2C`` and ``FMA2CVAL``.
    """
    fma2c_config = _resolve_fma_config('FMA2C', signals)
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    fringe_arrivals = dict()
    liquidity = dict()
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation['arrivals']
        liquidity[manager] += (len(signal.full_observation['departures']) - len(signal.full_observation['arrivals']))
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]['vehicles']:
                    if vehicle['id'] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = dict()
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += (fma2c_config['alpha'] * (fringe_arrivals[neighbor] + liquidity[neighbor]))
        management_neighborhood[manager] = mgr_rew

    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['FMA2C']['coef'])
        rewards[signal_id] = -reward

    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += (fma2c_config['alpha'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards


def fma2c_full(signals):
    """Full hierarchical reward variant for FMA2CFull experiments.

    Same reward structure as :func:`fma2c` (worker + manager terms), but with
     parameters sourced from ``mdp_configs['FMA2CFull']``. This allows running a
     separate experimental configuration without changing reward logic.

    Used by ``FMA2CFull``.
    """
    fma2c_config = _resolve_fma_config('FMA2CFull', signals)
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    fringe_arrivals = dict()
    liquidity = dict()
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation['arrivals']
        liquidity[manager] += (len(signal.full_observation['departures']) - len(signal.full_observation['arrivals']))
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]['vehicles']:
                    if vehicle['id'] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = dict()
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += (fma2c_config['alpha'] * (fringe_arrivals[neighbor] + liquidity[neighbor]))
        management_neighborhood[manager] = mgr_rew

    rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]['queue']
            reward += (signal.full_observation[lane]['max_wait'] * mdp_configs['FMA2CFull']['coef'])
        rewards[signal_id] = -reward

    neighborhood_rewards = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += (fma2c_config['alpha'] * rewards[neighbor])
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards

#endregion
#============================================================================================