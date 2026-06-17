import numpy as np

from TREX_comp.config.mdp_config import mdp_configs


#region Resolve FMA config

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

#endregion
#============================================================================================
#region DRQs

def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'])
            lane_obs.append(signal.full_observation[lane]['total_wait'])
            lane_obs.append(signal.full_observation[lane]['queue'])

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += vehicle['speed']
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'] / 28)
            lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
            lane_obs.append(signal.full_observation[lane]['queue'] / 28)

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_delta_norm(signals):
    """Alias for :func:`drq_norm` to pair with delta rewards."""
    return drq_norm(signals)

# NEW MULTIMODAL
_LANE_CAPACITY = 28
_MAX_SPEED_MS  = 20


def drq_mm2_delta(signals):
    """Multimodal DRQ observation with delta car/bike wait features."""
 
    # --- per-call previous-wait store (reset-safe) ---
    prev = getattr(drq_mm2_delta, '_prev', None)
    signal_ids = set(signals.keys())
    if prev is None or set(prev.keys()) != signal_ids:
        prev = {}
        drq_mm2_delta._prev = prev
 
    observations = {}
 
    for signal_id, signal in signals.items():
        active_lanes = set(signal.phase_lanes.get(signal.phase, []))
 
        ped_total_wait = float(signal.full_observation.get('ped_total_wait', 0.0)) / _LANE_CAPACITY
        ped_waiting    = float(signal.full_observation.get('ped_waiting',    0.0)) / _LANE_CAPACITY
 
        # Previous per-lane waits for this signal (dict keyed by lane id)
        prev_signal = prev.get(signal_id, {})
        next_prev_signal = {}
 
        obs = []
        for lane in signal.lanes:
            lm = signal.full_observation[lane]
 
            total_wait = float(lm.get('total_wait',      0.0))
            bike_wait  = float(lm.get('bike_total_wait', 0.0))
            car_wait   = max(0.0, total_wait - bike_wait)
 
            # Delta vs previous step; 0.0 on first step
            prev_car, prev_bike = prev_signal.get(lane, (car_wait, bike_wait))
            delta_car  = car_wait  - prev_car
            delta_bike = bike_wait - prev_bike
            next_prev_signal[lane] = (car_wait, bike_wait)
 
            vehicles   = lm.get('vehicles', [])
            n_vehicles = len(vehicles)
            mean_speed = (
                sum(float(v.get('speed', 0.0)) for v in vehicles) / n_vehicles / _MAX_SPEED_MS
                if n_vehicles > 0 else 0.0
            )
 
            obs.append([
                1.0 if lane in active_lanes else 0.0,
                float(lm.get('approach',   0.0)) / _LANE_CAPACITY,
                delta_car  / _LANE_CAPACITY,   # signed: negative = improvement
                delta_bike / _LANE_CAPACITY,   # signed: negative = improvement
                float(lm.get('queue',      0.0)) / _LANE_CAPACITY,
                float(lm.get('bike_queue', 0.0)) / _LANE_CAPACITY,
                mean_speed,
                ped_total_wait,
                ped_waiting,
            ])
 
        prev[signal_id] = next_prev_signal
 
        observations[signal_id] = np.expand_dims(
            np.asarray(obs, dtype=np.float32), axis=0
        )
 
    return observations

 
def _reset_drq_mm2_delta():
    drq_mm2_delta._prev = {}
 
 
drq_mm2_delta.reset = _reset_drq_mm2_delta

# First MM2 implementation
def drq_mm2(signals):
    """DRQ observation with car, bike, and pedestrian features.

    Shape per signal: (1, n_lanes, 9)

    Features per lane:
        active_phase   — 1 if this lane is actively served by the current
                            green phase (via Signal.phase_lanes), 0 otherwise
                            and during yellow transitions.
        approach       — approaching (non-waiting) vehicle count / LANE_CAPACITY
        car_wait       — (total_wait - bike_wait) / LANE_CAPACITY
        bike_wait      — bike_total_wait / LANE_CAPACITY
        queue          — waiting vehicle count / LANE_CAPACITY
        bike_queue     — waiting bike count / LANE_CAPACITY
        mean_speed     — mean vehicle speed / MAX_SPEED_MS (0.0 if lane empty)
        ped_total_wait — signal-level ped_total_wait / LANE_CAPACITY (repeated per lane)
        ped_waiting    — signal-level ped_waiting count / LANE_CAPACITY (repeated per lane)
    """
    observations = {}

    for signal_id, signal in signals.items():
        # Lanes actively served by the current phase; empty set during yellows
        # (signal.phase >= signal.num_green_phases has no phase_lanes entry).
        active_lanes = set(signal.phase_lanes.get(signal.phase, []))

        ped_total_wait = float(signal.full_observation.get('ped_total_wait', 0.0)) / _LANE_CAPACITY
        ped_waiting    = float(signal.full_observation.get('ped_waiting',    0.0)) / _LANE_CAPACITY

        obs = []
        for lane in signal.lanes:
            lane_measures = signal.full_observation[lane]

            total_wait = float(lane_measures.get('total_wait',      0.0))
            bike_wait  = float(lane_measures.get('bike_total_wait', 0.0))
            car_wait   = max(0.0, total_wait - bike_wait)

            vehicles   = lane_measures.get('vehicles', [])
            n_vehicles = len(vehicles)
            mean_speed = (
                sum(float(v.get('speed', 0.0)) for v in vehicles) / n_vehicles / _MAX_SPEED_MS
                if n_vehicles > 0 else 0.0
            )

            lane_obs = [
                1 if lane in active_lanes else 0,
                float(lane_measures.get('approach',   0.0)) / _LANE_CAPACITY,
                car_wait  / _LANE_CAPACITY,
                bike_wait / _LANE_CAPACITY,
                float(lane_measures.get('queue',      0.0)) / _LANE_CAPACITY,
                float(lane_measures.get('bike_queue', 0.0)) / _LANE_CAPACITY,
                mean_speed, # declared earlier
                ped_total_wait,
                ped_waiting,
            ]
            obs.append(lane_obs)

        observations[signal_id] = np.expand_dims(
            np.asarray(obs, dtype=np.float32), axis=0
        )

    return observations

# OLD STATE
# TODO: figure out good normalization
def drq_multimodal_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase # problem as lane index can be =! phase index

        # Pedestrian measures are signal-level; repeat per lane so tensor shape
        #  remains lane x feature and stays compatible with the existing IDQN model.
        # TODO: consider scales in mdp config instead of /28 here
        ped_total_wait = float(signal.full_observation.get('ped_total_wait', 0.0)) / 28
        ped_waiting = float(signal.full_observation.get('ped_waiting', 0.0)) / 28

        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_measures = signal.full_observation[lane]
            total_wait = float(lane_measures.get('total_wait', 0.0))
            bike_wait = float(lane_measures.get('bike_total_wait', 0.0))
            car_wait = max(0.0, total_wait - bike_wait)

            lane_obs.append(float(lane_measures.get('approach', 0.0)) / 28)
            lane_obs.append(car_wait / 28)
            lane_obs.append(bike_wait / 28)
            lane_obs.append(float(lane_measures.get('queue', 0.0)) / 28)
            lane_obs.append(float(lane_measures.get('bike_queue', 0.0)) / 28)

            total_speed = 0.0
            vehicles = lane_measures.get('vehicles', [])
            for vehicle in vehicles:
                total_speed += float(vehicle.get('speed', 0.0)) / 20 / 28
            lane_obs.append(total_speed)

            lane_obs.append(ped_total_wait)
            lane_obs.append(ped_waiting)

            obs.append(lane_obs)

        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations

#endregion
#============================================================================================
#region MPLights

def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += (signal.full_observation[lane]['total_wait'] / 28)
                total_speed = 0
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                tot_approach += (signal.full_observation[lane]['approach'] / 28)

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations

def mplight_mm(signals, num_phase_pairs=None):
    """MPLight state function extended with bike and pedestrian demand.

    num_phase_pairs: the GLOBAL len(phase_pairs) for this map (same value
    FRAP_MM uses as output_shape/num_actions). This must be passed in
    (e.g. via functools.partial when wiring up agt_config['state']) rather
    than inferred per-signal, because MPLight_MM is a SharedAgent: every
    signal's state vector is batched together and must be the same length,
    even though individual intersections may not use every phase pair
    (signal_configs[map]['valid_acts'][signal_id] can be a subset).

    Per signal, the observation vector is laid out as:

        [phase,
         m0_car_pressure, m0_bike_pressure,
         m1_car_pressure, m1_bike_pressure,
         ...,
         p0_ped_pressure, p1_ped_pressure, ..., p{N-1}_ped_pressure]

    i.e. for each movement (direction in signal.lane_sets) we emit a pair
    (car_pressure, bike_pressure) — this is what lets FRAP_MM use
    demand_shape=2 and otherwise reuse the unmodified per-movement FRAP
    pipeline. After all movements, we append ONE pedestrian-pressure scalar
    per phase_pair index (zero-padded to max_phase_pairs across signals,
    same way num_actions is shared/padded today via valid_acts), so the
    model can bias phase-pair scores with FRAP_MM's pair-level pedestrian
    embedding.

    car_pressure / bike_pressure follow the same inbound-minus-downstream
    logic as the original mplight() state fn, just split by vehicle class
    using the 'queue' vs 'bike_queue' lane fields already populated in
    Signal.observe().

    ped_pressure for phase_pair i = sum of signal.ped_crossing_pressure[d]
    for every crossing direction d served during that phase pair (per
    Signal._build_phase_pair_ped_crossings), i.e. "how much pedestrian
    demand would get served if we picked this phase pair right now."

    Requires Signal to expose (see signals_mm_patch.py):
        - signal.phase_pair_ped_crossings: dict[pair_idx -> list[direction]]
        - signal.ped_crossing_pressure: dict[direction -> float], refreshed
          each observe() call.
    """
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]

        for direction in signal.lane_sets:
            car_pressure = 0
            bike_pressure = 0
            for lane in signal.lane_sets[direction]:
                lane_obs = signal.full_observation[lane]
                total_queue = lane_obs['queue']
                bike_queue = lane_obs.get('bike_queue', 0)
                car_pressure += (total_queue - bike_queue)
                bike_pressure += bike_queue

            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    dwn_obs = signal.signals[dwn_signal].full_observation[lane]
                    dwn_total = dwn_obs['queue']
                    dwn_bike = dwn_obs.get('bike_queue', 0)
                    car_pressure -= (dwn_total - dwn_bike)
                    bike_pressure -= dwn_bike

            obs.append(car_pressure)
            obs.append(bike_pressure)

        # Pedestrian context, one scalar per GLOBAL phase_pair index.
        # Phase pairs this signal doesn't actually use (not in its
        # valid_acts) stay zero — same padding convention the rest of
        # MPLight already relies on for the action/Q-value head.
        if num_phase_pairs is None:
            raise ValueError(
                "mplight_mm requires num_phase_pairs (global len(phase_pairs) "
                "for this map) — wire it via functools.partial in agent_config, "
                "e.g. partial(mplight_mm, num_phase_pairs=len(signal_configs[map]['phase_pairs']))"
            )
        ped_block = np.zeros(num_phase_pairs, dtype=np.float32)
        crossing_pressure = getattr(signal, 'ped_crossing_pressure', {})
        for pair_idx, directions in getattr(signal, 'phase_pair_ped_crossings', {}).items():
            if pair_idx >= num_phase_pairs:
                continue
            ped_block[pair_idx] = sum(crossing_pressure.get(d, 0.0) for d in directions)

        obs.extend(ped_block.tolist())
        observations[signal_id] = np.asarray(obs, dtype=np.float32)
    return observations

#endregion
#============================================================================================
#region Wave

def wave(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        state = []
        for direction in signal.lane_sets:
            wave_sum = 0
            for lane in signal.lane_sets[direction]:
                wave_sum += signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            state.append(wave_sum)
        observations[signal_id] = np.asarray(state)
    return observations

#endregion
#============================================================================================
#region MA2C & FMA2C

def ma2c(signals):
    ma2c_config = mdp_configs['MA2C']

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / ma2c_config['norm_wave'], 0, ma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(ma2c_config['coop_gamma'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / ma2c_config['norm_wait'], 0, ma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    fma2c_config = _resolve_fma_config('FMA2C', signals)
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    fma2c_config = _resolve_fma_config('FMA2CFull', signals)
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)

            waves.append(signal.full_observation[lane]['total_wait'] / 28)
            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations