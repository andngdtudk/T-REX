import numpy as np

from TREX_comp.config.mdp_config import mdp_configs


def _resolve_fma_config(config_key, signals):
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


def wait(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = -total_wait
    return rewards


def wait_norm(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = np.clip(-total_wait/224, -4, 4).astype(np.float32)
    return rewards


def pressure(signals):
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


def fma2c(signals):
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