import csv
from time import sleep
from unittest import signals

import numpy as np
import csv

from TREX_comp.config.mdp_config import mdp_configs
import traci

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
    """Compute raw delta waits and update previous-wait state.

    Returns
        rewards : dict[signal_id, float]
            Positive values mean waiting time decreased (improvement).
            Zero on the first step for any signal not yet seen.
    """
    rewards = {}

    for signal_id, signal in signals.items():
        total_wait = sum(
            signal.full_observation[lane].get('total_wait', 0.0)
            for lane in signal.lanes
        )

        prev_wait = prev_waits.get(signal_id)
        rewards[signal_id] = 0.0 if prev_wait is None else float(prev_wait - total_wait)
        prev_waits[signal_id] = total_wait

    # Log all signals in a single pass, one open() per call
    with open('logs/delta_log.csv', 'a') as f:
        writer = csv.writer(f)
        for signal_id, val in rewards.items():
            writer.writerow([signal_id, val])

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
        rewards[signal_id] = np.float32(
            np.clip(value / cfg['norm_wait'],
            -cfg['clip_wait'],
            cfg['clip_wait'],
            )
        )
    return rewards

# For variance version
def _get_delta_var_config(config_key, require_clip=False):
    raw = mdp_configs.get(config_key)
    if not isinstance(raw, dict):
        raise ValueError(f"Missing mdp_configs['{config_key}'] for delta fairness config")

    if 'norm_wait' not in raw:
        raise ValueError(f"Missing norm_wait in mdp_configs['{config_key}']")
    if 'lambda_f' not in raw:
        raise ValueError(f"Missing lambda_f in mdp_configs['{config_key}']")
    if require_clip and 'clip_wait' not in raw:
        raise ValueError(f"Missing clip_wait in mdp_configs['{config_key}']")

    cfg = {
        'norm_wait': raw['norm_wait'],
        'clip_wait': raw.get('clip_wait'),
        'lambda_f': raw['lambda_f'],
    }
    return cfg


def wait_delta_var(signals):
    """Delta wait reward with variance-based fairness penalty for IDQN_DELTAVAR."""
    cfg = _get_delta_var_config('IDQN_DELTAVAR', require_clip=True)
    prev_waits = getattr(wait_delta_var, '_prev_waits', None)
    signal_ids = set(signals.keys())
    if prev_waits is None or set(prev_waits.keys()) != signal_ids:
        prev_waits = {}
        wait_delta_var._prev_waits = prev_waits

    rewards = {}
    """ print(f"Total controlled lanes: {len(control_lanes)}")
    control_lanes = traci.trafficlight.getControlledLanes('J8')
    for i, lane in enumerate(control_lanes):
        print(f"Position {i}: {lane}")"""

    for signal_id, signal in signals.items():
        """ print("Lanes in agent observation:")
        for lane in signals[signal_id].lanes:
            print(f"  {lane}")
            sleep(5)  # Add a small delay to ensure all lane information is printed """
        total_wait = 0.0
        lane_waits = []
        for lane in signal.lanes:
            lane_wait = float(signal.full_observation[lane]['total_wait'])
            total_wait += lane_wait
            lane_waits.append(lane_wait)

        prev_wait = prev_waits.get(signal_id)
        if prev_wait is None:
            delta_wait = 0.0
        else:
            delta_wait = prev_wait - total_wait
        prev_waits[signal_id] = total_wait

        mean_wait = np.mean(lane_waits) if lane_waits else 1.0
        cv = np.std(lane_waits) / (mean_wait + 1e-8)  # coefficient of variation
        if lane_waits:
            fairness_penalty = cfg['lambda_f'] *  cv * mean_wait
        else:
            fairness_penalty = 0.0
        reward = (delta_wait - fairness_penalty) / cfg['norm_wait']
        rewards[signal_id] = np.clip(
            reward,
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


def _reset_wait_delta_var():
    wait_delta_var._prev_waits = {}


wait_delta_scale.reset = _reset_wait_delta_scale
wait_delta_sclip.reset = _reset_wait_delta_sclip
wait_delta_var.reset = _reset_wait_delta_var

#endregion
#============================================================================================
#region Wait multimodal

def _resolve_multimodal_delta_config():
    """Resolve config for multimodal delta waiting-time rewards (IDQN_MM2).

    Lookup order:
        1. mdp_configs['IDQN_MM2']
        2. mdp_configs['IDQN_DELTASCLIP']
        3. Hard-coded defaults

    Required keys: norm_wait, clip_wait, car_wait_weight, bike_wait_weight, ped_wait_weight
    """
    defaults = {
        'car_wait_weight': 1.0,
        'bike_wait_weight': 1.0,
        'ped_wait_weight': 1.0,
        'norm_wait': 224.0,
        'clip_wait': 4.0,
    }

    for key in ('IDQN_MM2', 'IDQN_DELTASCLIP', 'IDQN_MULTIMODAL', 'IDQN'):
        raw = mdp_configs.get(key)
        if not isinstance(raw, dict):
            continue
        has_modal_weights = all(
            k in raw for k in ('car_wait_weight', 'bike_wait_weight', 'ped_wait_weight')
        )
        has_delta_params = 'norm_wait' in raw and 'clip_wait' in raw
        if has_modal_weights and has_delta_params:
            cfg = defaults.copy()
            cfg.update(raw)
            return cfg

    return defaults


def wait_multimodal_delta_sclip(signals):
    """Scaled + clipped delta of weighted multimodal wait reward (IDQN_MM2).

    Computes the *change* in combined waiting time per signal between the
    current and previous step, then normalises and clips:

        combined_wait(t) =
            car_wait_weight  * car_wait(t)
            + bike_wait_weight  * bike_wait(t)
            + ped_wait_weight  * ped_wait(t)

        reward = clip(
            -(combined_wait(t) - combined_wait(t-1)) / norm_wait,
            -clip_wait,
            +clip_wait,
        )

    A positive reward means the combined wait *decreased* (improvement).

    Wait components:
        car_wait  — sum over lanes of max(0, total_wait - bike_total_wait)
        bike_wait — sum over lanes of bike_total_wait
        ped_wait  — signal-level ped_total_wait

    Config source (first match wins):
        mdp_configs['IDQN_MM2']
        mdp_configs['IDQN_DELTASCLIP']
        mdp_configs['IDQN_MULTIMODAL']
        mdp_configs['IDQN']
        hard-coded defaults (norm_wait=224.0, clip_wait=4.0, all weights=1.0)
    """
    cfg = _resolve_multimodal_delta_config()

    # --- initialise / validate persistent wait store ---
    prev_waits = getattr(wait_multimodal_delta_sclip, '_prev_waits', None)
    signal_ids = set(signals.keys())
    if prev_waits is None or set(prev_waits.keys()) != signal_ids:
        prev_waits = {}
        wait_multimodal_delta_sclip._prev_waits = prev_waits

    # --- compute current combined waits ---
    rewards = {}
    next_prev_waits = {}

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
        next_prev_waits[signal_id] = combined_wait

        # On the very first step there is no previous wait, so reward is 0.
        prev_combined = prev_waits.get(signal_id, combined_wait)
        delta = combined_wait - prev_combined  # positive = got worse

        rewards[signal_id] = np.float32(
            np.clip(-delta / cfg['norm_wait'], -cfg['clip_wait'], cfg['clip_wait'])
        )

    wait_multimodal_delta_sclip._prev_waits = next_prev_waits
    return rewards


def _reset_wait_multimodal_delta_sclip():
    """Reset persistent state between episodes."""
    wait_multimodal_delta_sclip._prev_waits = {}

wait_multimodal_delta_sclip.reset = _reset_wait_multimodal_delta_sclip

# OLD METHODS
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

_PRESSURE_LOG_PATH = "pressure_log.csv"
_pressure_log_initialized = False

def _log_pressures(signal_id, car_pressure, bike_pressure, ped_pressure, step):
    global _pressure_log_initialized
    mode = 'a' if _pressure_log_initialized else 'w'
    with open(_PRESSURE_LOG_PATH, mode, newline='') as f:
        writer = csv.writer(f)
        if not _pressure_log_initialized:
            writer.writerow(['step', 'signal_id', 'car_pressure', 'bike_pressure', 'ped_pressure'])
            _pressure_log_initialized = True
        writer.writerow([step, signal_id, car_pressure, bike_pressure, ped_pressure])

def mplight_mm(signals, sim_time):
    """Traffic-pressure reward extended with bike and pedestrian pressure.
 
    reward = -(car_pressure + W_BIKE * bike_pressure + W_PED * ped_pressure)
 
    car_pressure / bike_pressure: same inbound-minus-downstream queue
    pressure as the original mplight reward, split by vehicle class using
    the 'queue' / 'bike_queue' lane fields.
 
    ped_pressure: sum over all of this signal's phase pairs of
    signal.ped_crossing_pressure[pair_idx] — SUMO's own
    traci.trafficlight.getServedPersonCount() per phase (see
    Signal._collect_ped_crossing_pressure), already keyed by GLOBAL
    phase_pairs index, not a crossing id. This replaced three earlier
    designs (cardinal-direction bucketing, real-edge path tracing,
    walking-area presence polling) that each had confirmed problems on
    real network topology/pedestrian-model behavior; getServedPersonCount
    is SUMO's own built-in answer to "how many people would be served by
    this phase," so this code no longer does any crossing detection
    itself. Unlike the vehicle terms there's no "downstream signal" to
    subtract for pedestrians — crossing a leg of THIS intersection
    doesn't create pressure at the next intersection the way a vehicle
    queue does, so we don't apply the same upstream-minus-downstream
    logic here.
    """
    rewards = dict()

    # We import weights from agent_config.py
    W_BIKE = mdp_configs['MPLight_MM']['W_BIKE']
    W_PED = mdp_configs['MPLight_MM']['W_PED']
    PED_NORM = mdp_configs['MPLight_MM']['PED_NORM']

    for signal_id in signals:
        signal = signals[signal_id]

        car_pressure = 0.0
        bike_pressure = 0.0
        for lane in signal.lanes:
            lane_obs = signal.full_observation[lane]
            total_queue = lane_obs['queue']
            bike_queue = lane_obs.get('bike_queue', 0)
            car_pressure += (total_queue - bike_queue)
            bike_pressure += bike_queue

        for lane in signal.outbound_lanes:
            dwn_signal = signal.out_lane_to_signalid[lane]
            if dwn_signal in signal.signals:
                dwn_obs = signal.signals[dwn_signal].full_observation[lane]
                dwn_total = dwn_obs['queue']
                dwn_bike = dwn_obs.get('bike_queue', 0)
                car_pressure -= (dwn_total - dwn_bike)
                bike_pressure -= dwn_bike

        ped_pressure_raw = sum(getattr(signal, 'ped_crossing_pressure', {}).values())
        # Clip BEFORE weighting — a rare crowd-crossing event at one signal
        # shouldn't be able to produce a TD error far outside what car/bike
        # pressure ever produces. PED_NORM=22 was derived as the typical
        # (p90-ish) magnitude; clip here uses it as a ceiling, not a divisor,
        # so typical values pass through unchanged and only the tail is capped.
        ped_pressure = min(ped_pressure_raw, PED_NORM)

        _log_pressures(signal_id, car_pressure, bike_pressure, ped_pressure_raw, sim_time)

        rewards[signal_id] = -(car_pressure + W_BIKE * bike_pressure + W_PED * ped_pressure)
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