import os
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import deque
import csv
import numpy as np

from TREX_comp.config.agent_config import agent_configs
from TREX_comp.config.map_config import map_configs
from TREX_comp.config.mdp_config import mdp_configs

from incident_env import IncidentEnv
from base_env import BaseEnv


AGENT_ALIASES = {
    'FMA2CFull': 'FMA2CFULL',
    "IDQNMULTI": "IDQN_MULTIMODAL",
    "IDQN_MULTI": "IDQN_MULTIMODAL",
    "MPLIGHTFULL": "MPLightFULL",
    "MPLIGHT": "MPLight",
    "IDQN_D": "IDQN_DELTASCLIP",
}



def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run traffic signal control simulations with different agents.")
    
    parser.add_argument("--agent", type=str, default='MPLight',
                        choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE', 'IDQN', "IDQN_MULTIMODAL", 'IDQN_DELTA',
                                 'IDQN_DELTASCALE', 'IDQN_DELTASCLIP', 'IDQN_DELTAVAR', 'IPPO', 'MPLight', 'MA2C', 'FMA2C',
                                 'MPLightFULL', 'FMA2CFull', 'FMA2CVAL'],
                        help="Choose the RL-based or rule-based agent to run.")
                        # TODO: fix MA2C
    parser.add_argument("--map", type=str, default='grid4x4',
                        choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                                 'cologne1', 'cologne3', 'cologne8', "kbh_red", "kbh_red_2", "kbh_red_3",
                                 "kbh_red_4", "kbh_red_5","kbh_full", "kbh_full_multimodal", "kbh_full_multimodal_mod",
                                 "kbh_red_420", "kbh_joined_432", "kbh_red_430", "kbh_single_432", "kbh_j1_441"],
                        help="Specify the traffic network map.")

    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--eps", type=int, default=100, help="Number of episodes per trial.")
    parser.add_argument("--procs", type=int, default=1, help="Number of parallel processes to use.")

    parser.add_argument("--pwd", type=str, default=os.path.dirname(__file__), help="Project working directory.")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'results' + os.sep),
                        help="Directory to save logs and results.")
    parser.add_argument("--load_dir", type=str, default=None,
                        help="Directory to load a saved model from (overrides log_dir when --load is True).")

    parser.add_argument("--gui", type=bool, default=False, help="Whether to enable SUMO GUI.")
    parser.add_argument("--libsumo", dest="libsumo", action="store_true",
                        help="Use libsumo instead of traci.")
    parser.add_argument("--no-libsumo", dest="libsumo", action="store_false",
                        help="Disable libsumo and use traci.")
    parser.set_defaults(libsumo=True)
    parser.add_argument("--tr", type=int, default=0, help="Trial number (important when using libsumo).")

    parser.add_argument("--seps", type=int, default=0, help="Episode to start from when resuming training.")
    parser.add_argument("--load", type=bool, default=False, help="Whether to load a saved model.")

    parser.add_argument("--strategy", type=int, default=2,
                        help="Training strategy: 1 = base, 2 = incident, 3 = curriculum.")
    parser.add_argument("--repeat", type=int, default=0,
                        help="How many episodes to repeat incidents from training in testing.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the agent.")
    parser.add_argument(
        "--max_green_hold",
        type=int,
        default=12,
        help="Maximum consecutive control decisions to keep the same green phase before forcing a switch.",
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.libsumo and not os.environ.get('LIBSUMO_AS_TRACI'):
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to a nonempty value to enable libsumo.")

    if args.procs == 1 or args.libsumo:
        run_trial(args, args.tr)
    else:
        with mp.Pool(processes=args.procs) as pool:
            for trial in range(1, args.trials + 1):
                pool.apply_async(run_trial, args=(args, trial))
            pool.close()
            pool.join()

def run_trial(args, trial):
    # === Load Configurations ===
    agent_key = AGENT_ALIASES.get(args.agent, args.agent)
    mdp_key = args.agent if args.agent in mdp_configs else agent_key

    mdp_config = mdp_configs.get(mdp_key, {}).get(args.map)
    agt_config = agent_configs[agent_key]
    map_config = map_configs[args.map]

    if mdp_config:
        mdp_configs[mdp_key] = mdp_config
        agt_config['mdp'] = mdp_config
        management = mdp_config.get('management')
        if management:
            # Precompute supervisors (reverse mapping)
            mdp_config['supervisors'] = {
                worker: manager
                for manager, workers in management.items()
                for worker in workers
            }

    # === Environment Setup ===
    route = os.path.join(args.pwd, map_config['route']) if map_config.get('route') else None
    if args.map in {'grid4x4', 'arterial4x4'} and not os.path.exists(route):
        raise EnvironmentError("Please decompress the traffic flow files for the selected map.")

    env_class = BaseEnv if args.strategy == 1 else IncidentEnv
    env = env_class(
        run_name=f"{agent_key}-tr{trial}",
        map_name=args.map,
        net=os.path.join(args.pwd, map_config['net']),
        state_fn=agt_config['state'],
        reward_fn=agt_config['reward'],
        route=route,
        step_length=map_config['step_length'],
        yellow_length=map_config['yellow_length'],
        step_ratio=map_config['step_ratio'],
        end_time=map_config['end_time'],
        max_distance=agt_config['max_distance'],
        lights=map_config['lights'],
        gui=args.gui,
        log_dir=args.log_dir,
        libsumo=args.libsumo,
        warmup=map_config['warmup'],
        run=args.seps,
        level=2 if args.strategy == 2 else None,
        max_green_hold_steps=args.max_green_hold,
    )

    if mdp_key in {'FMA2C', 'FMA2CFull', 'FMA2CVAL'}:
        runtime_mdp = agt_config.get('mdp', {})
        if 'management' not in runtime_mdp:
            manager = 'top_mgr'
            runtime_mdp = {
                'coef': 0.4,
                'coop_gamma': 0.9,
                'clip_wave': 4.0,
                'clip_wait': 4.0,
                'norm_wave': 5.0,
                'norm_wait': 100.0,
                'alpha': 0.75,
                'management': {manager: list(env.all_ts_ids)},
                'management_neighbors': {manager: []},
            }

        management = runtime_mdp.get('management', {})
        runtime_mdp['supervisors'] = {
            worker: mgr
            for mgr, workers in management.items()
            for worker in workers
        }
        default_manager = next(iter(management), 'top_mgr')
        for signal_id in env.all_ts_ids:
            runtime_mdp['supervisors'].setdefault(signal_id, default_manager)

        agt_config['mdp'] = runtime_mdp
        mdp_configs[mdp_key] = runtime_mdp

    # === Agent Setup ===
    alg = agt_config['agent']
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    train_eps = max(1, int(args.eps * 0.8))
    train_steps = max(1, train_eps * num_steps_eps)
    run_log_dir = os.path.join(args.log_dir, env.connection_name)
    load_dir = args.load_dir if args.load_dir else run_log_dir
    agt_config.update({
        'episodes': train_eps,
        'steps': train_steps,
        'log_dir': run_log_dir,
        'load_dir': load_dir,
        'num_lights': len(env.all_ts_ids),
        'save_freq': 50 if alg.__name__ in {'IPPO', 'FMA2C'} else 10,
        'load': args.load
    })

    obs_act = {
        key: [env.obs_shape[key], len(env.phases.get(key, []))]
        for key in env.obs_shape
    }

    agent = alg(agt_config, obs_act, args.map, trial, lr=args.lr) if alg.__name__ in {'MPLight', 'IDQN'} else \
            alg(agt_config, obs_act, args.map, trial)

    feature_names = _get_idqn_feature_names(env.state_fn)
    log_state_csv = None
    if alg.__name__ == 'IDQN' and args.map == 'kbh_joined_432' and feature_names:
        csv_path = os.path.join(agt_config['log_dir'], 'state_feature_stats.csv')
        log_state_csv = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(
            log_state_csv,
            fieldnames=[
                'episode', 'decision', 'feature', 'count', 'mean', 'std', 'min', 'max',
                'p05', 'p95', 'nan_count', 'inf_count', 'zero_frac'
            ],
        )
        writer.writeheader()
        log_state_csv = (log_state_csv, writer, feature_names)

    # === Training or Testing ===
    try:
        if args.strategy == 1:
            run_base_scenario(env, agent, args, agt_config, log_state_csv)
        else:
            run_incident_scenario(env, agent, args, agt_config, alg, log_state_csv)
    finally:
        if log_state_csv is not None:
            log_state_csv[0].close()
        env.close()
        print(f"Trial {trial} completed for {args.agent} on {args.map}.", flush=True)


# === Helper Functions ===

def run_base_scenario(env, agent, args, agt_config, log_state_csv):
    if agt_config['load']:
        print('Testing under base condition...')
        for ep in range(args.seps, args.eps):
            print(f"Episode {ep + 1}/{args.eps} started (base test).", flush=True)
            stats = run_episode(env, agent, episode_index=ep + 1, log_state_csv=log_state_csv)
            print(
                f"Episode {ep + 1}/{args.eps} finished: decisions={stats['decisions']}, "
                f"sim_time={stats['sim_time']:.1f}, done={stats['done']}",
                flush=True,
            )
    else:
        print('Training under base condition...')
        for ep in range(args.eps):
            print(f"Episode {ep + 1}/{args.eps} started (base train).", flush=True)
            stats = run_episode(env, agent, episode_index=ep + 1, log_state_csv=log_state_csv)
            print(
                f"Episode {ep + 1}/{args.eps} finished: decisions={stats['decisions']}, "
                f"sim_time={stats['sim_time']:.1f}, done={stats['done']}",
                flush=True,
            )


def run_incident_scenario(env, agent, args, agt_config, alg, log_state_csv):
    map_id = args.map
    agent_name = alg.__name__
    seed_file_1 = f"{agent_name}{map_id}-seed_ic1.txt"
    seed_file_2 = f"{agent_name}{map_id}-seed_ic2.txt"

    if agt_config['load']:
        print('Testing under incident condition...')
        if args.repeat > 0 and os.path.exists(seed_file_1) and os.path.exists(seed_file_2):
            last_seeds_ic1 = load_seeds(seed_file_1, args.repeat)
            last_seeds_ic2 = load_seeds(seed_file_2, args.repeat)
            print(f"Loaded seeds from files: {list(last_seeds_ic1)}, {list(last_seeds_ic2)}")

            for ep in range(args.seps, args.eps):
                seed_ic1, seed_ic2 = last_seeds_ic1.popleft(), last_seeds_ic2.popleft()
                obs = env.reset(pre_seed=[seed_ic1, seed_ic2])
                print(f"Episode {ep + 1}/{args.eps} started (incident test, seeded).", flush=True)
                stats = run_episode(env, agent, obs, episode_index=ep + 1, log_state_csv=log_state_csv)
                print(
                    f"Episode {ep + 1}/{args.eps} finished: decisions={stats['decisions']}, "
                    f"sim_time={stats['sim_time']:.1f}, done={stats['done']}",
                    flush=True,
                )
        else:
            print('Testing without predefined incident seeds...')
            for ep in range(args.seps, args.eps):
                print(f"Episode {ep + 1}/{args.eps} started (incident test).", flush=True)
                stats = run_episode(env, agent, episode_index=ep + 1, log_state_csv=log_state_csv)
                print(
                    f"Episode {ep + 1}/{args.eps} finished: decisions={stats['decisions']}, "
                    f"sim_time={stats['sim_time']:.1f}, done={stats['done']}",
                    flush=True,
                )
    else:
        print('Training under incident condition...')
        if args.repeat > 0:
            print('Training and saving last incident seeds...')
            last_seed_ic1 = deque(maxlen=args.repeat)
            last_seed_ic2 = deque(maxlen=args.repeat)

            for ep in range(args.eps):
                obs = env.reset()
                if ep >= args.eps - args.repeat:
                    last_seed_ic1.append(env.seed_ic1)
                    last_seed_ic2.append(env.seed_ic2)
                print(f"Episode {ep + 1}/{args.eps} started (incident train, seed capture).", flush=True)
                stats = run_episode(env, agent, obs, episode_index=ep + 1, log_state_csv=log_state_csv)
                print(
                    f"Episode {ep + 1}/{args.eps} finished: decisions={stats['decisions']}, "
                    f"sim_time={stats['sim_time']:.1f}, done={stats['done']}",
                    flush=True,
                )

            save_seeds(seed_file_1, last_seed_ic1)
            save_seeds(seed_file_2, last_seed_ic2)
        else:
            print('Training without saving incident seeds...')
            for ep in range(args.eps):
                print(f"Episode {ep + 1}/{args.eps} started (incident train).", flush=True)
                stats = run_episode(env, agent, episode_index=ep + 1, log_state_csv=log_state_csv)
                print(
                    f"Episode {ep + 1}/{args.eps} finished: decisions={stats['decisions']}, "
                    f"sim_time={stats['sim_time']:.1f}, done={stats['done']}",
                    flush=True,
                )


def _sum_lane_metric(signals, metric_key):
    total = 0.0
    for signal in signals.values():
        for lane in signal.lanes:
            total += float(signal.full_observation[lane].get(metric_key, 0.0))
    return total


def _log_step_reward(env, rewards, step_number):
    reward_name = getattr(env.reward_fn, '__name__', '')
    if reward_name.startswith('fma2c'):
        print(f"  step {step_number}: reward logging not implemented", flush=True)
        return

    if reward_name == 'pressure':
        total_queue = _sum_lane_metric(env.signals, 'queue')
        print(f"  step {step_number}: total_queue={total_queue:.2f}", flush=True)
        return

    if reward_name in {'wait', 'wait_norm'}:
        total_wait = _sum_lane_metric(env.signals, 'total_wait')
        print(f"  step {step_number}: total_wait={total_wait:.2f}", flush=True)
        return

    if reward_name == 'wait_multimodal_norm':
        total_wait = _sum_lane_metric(env.signals, 'total_wait')
        ped_wait = 0.0
        for signal in env.signals.values():
            ped_wait += float(signal.full_observation.get('ped_total_wait', 0.0))
        print(f"  step {step_number}: total_wait={total_wait + ped_wait:.2f}", flush=True)
        return

    total_reward = 0.0
    if isinstance(rewards, dict):
        for value in rewards.values():
            total_reward += float(value)
    print(f"  step {step_number}: total_reward={total_reward:.4f}", flush=True)


def _log_step_metrics(agent, step_metrics, step_number):
    if step_metrics:
        q_mean = step_metrics.get("q_mean")
        q_max = step_metrics.get("q_max")
        q_min = step_metrics.get("q_min")
        entropy = step_metrics.get("action_entropy")
        print(
            f"  step {step_number}: q_mean={q_mean:.4f} q_max={q_max:.4f} q_min={q_min:.4f} "
            f"entropy={entropy:.4f}",
            flush=True,
        )

    if hasattr(agent, "training_stats"):
        stats = agent.training_stats()
        if stats:
            loss = stats.get("average_loss", stats.get("loss"))
            td_error = None
            for key in stats:
                if "td" in key.lower():
                    td_error = stats[key]
                    break
            avg_q = stats.get("average_q", stats.get("q"))

            parts = []
            if loss is not None:
                parts.append(f"loss={loss:.6f}")
            if td_error is not None:
                parts.append(f"td_error={td_error:.6f}")
            if avg_q is not None:
                parts.append(f"avg_q={avg_q:.6f}")
            if parts:
                print(f"  step {step_number}: " + " ".join(parts), flush=True)


def run_episode(env, agent, obs=None, episode_index=None, log_state_csv=None):
    debug_episode = os.getenv('TREX_DEBUG_EPISODE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    if obs is None:
        obs = env.reset()
    done = False
    decisions = 0
    while not done:
        if debug_episode:
            print(f"  debug: decision {decisions + 1} -> act", flush=True)
        step_metrics = agent.step_metrics(obs) if hasattr(agent, "step_metrics") else None
        act = agent.act(obs)
        safe_act = {}
        for signal_id in getattr(env, 'signal_ids', []):
            selected = act.get(signal_id, 0)
            num_phases = len(env.phases.get(signal_id, []))
            if num_phases <= 0:
                safe_selected = 0
            else:
                safe_selected = int(selected) % num_phases
            safe_act[signal_id] = safe_selected
        act = safe_act
        if debug_episode:
            print(f"  debug: decision {decisions + 1} actions={act}", flush=True)
            print(f"  debug: decision {decisions + 1} -> step", flush=True)
        obs, rew, done, info = env.step(act)
        _log_state_features(obs, episode_index, decisions + 1, log_state_csv)
        _log_step_reward(env, rew, decisions + 1)
        if debug_episode:
            print(f"  debug: decision {decisions + 1} -> observe", flush=True)
        agent.observe(obs, rew, done, info)
        _log_step_metrics(agent, step_metrics, decisions + 1)
        decisions += 1
        if decisions % 100 == 0:
            print(
                f"  progress: decisions={decisions}, sim_time={env.sumo.simulation.getTime():.1f}",
                flush=True,
            )

    sim_time = env.sumo.simulation.getTime() if hasattr(env, 'sumo') else -1
    return {
        'decisions': decisions,
        'sim_time': sim_time,
        'done': done,
    }


def _get_idqn_feature_names(state_fn):
    state_name = getattr(state_fn, '__name__', '')
    if state_name in {'drq_norm', 'drq_delta_norm'}:
        return ['phase', 'approach', 'total_wait', 'queue', 'total_speed']
    if state_name == 'drq_multimodal_norm':
        return [
            'phase', 'approach', 'car_wait', 'bike_wait', 'queue', 'bike_queue',
            'total_speed', 'ped_total_wait', 'ped_waiting'
        ]
    return None


def _log_state_features(obs, episode_index, decision_index, log_state_csv):
    if log_state_csv is None:
        return
    file_handle, writer, feature_names = log_state_csv
    if not isinstance(obs, dict):
        return

    arrays = []
    expected_features = len(feature_names)
    for obs_item in obs.values():
        obs_arr = np.asarray(obs_item, dtype=np.float32)
        if obs_arr.ndim == 3:
            obs_arr = obs_arr[0]
        if obs_arr.ndim != 2 or obs_arr.shape[-1] != expected_features:
            continue
        arrays.append(obs_arr.reshape(-1, expected_features))

    if not arrays:
        return

    stacked = np.concatenate(arrays, axis=0)
    for idx, feature in enumerate(feature_names):
        values = stacked[:, idx]
        nan_mask = np.isnan(values)
        inf_mask = np.isinf(values)
        clean = values[~nan_mask & ~inf_mask]
        if clean.size == 0:
            stats = {
                'episode': episode_index,
                'decision': decision_index,
                'feature': feature,
                'count': 0,
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'p05': float('nan'),
                'p95': float('nan'),
                'nan_count': int(np.sum(nan_mask)),
                'inf_count': int(np.sum(inf_mask)),
                'zero_frac': float('nan'),
            }
        else:
            stats = {
                'episode': episode_index,
                'decision': decision_index,
                'feature': feature,
                'count': int(clean.size),
                'mean': float(np.mean(clean)),
                'std': float(np.std(clean)),
                'min': float(np.min(clean)),
                'max': float(np.max(clean)),
                'p05': float(np.percentile(clean, 5)),
                'p95': float(np.percentile(clean, 95)),
                'nan_count': int(np.sum(nan_mask)),
                'inf_count': int(np.sum(inf_mask)),
                'zero_frac': float(np.mean(clean == 0.0)),
            }
        writer.writerow(stats)


def load_seeds(filename, max_len):
    with open(filename, "r") as f:
        return deque([int(line.strip()) for line in f.readlines()], maxlen=max_len)


def save_seeds(filename, seeds):
    with open(filename, "w") as f:
        for seed in seeds:
            f.write(f"{seed}\n")
    print(f"Saved seeds to {filename}: {list(seeds)}")


if __name__ == "__main__":
    main()
