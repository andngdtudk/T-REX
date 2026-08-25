import os
import argparse
import logging
import multiprocessing as mp
import random
from collections import deque

import numpy as np
import torch

from TREX_comp.config.agent_config import agent_configs
from TREX_comp.config.map_config import map_configs
from TREX_comp.config.mdp_config import mdp_configs
from TREX_comp.config.hyperparams import resolve_learning_rate
from TREX_comp.config.incident_config import DEFAULT_INCIDENTS, NO_INCIDENTS

from trex_env import TrexEnv

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run traffic signal control simulations with different agents.")
    
    parser.add_argument("--agent", type=str, default='MPLight',
                        choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE', 'IDQN', 'IPPO',
                                 'MPLight', 'MA2C', 'FMA2C', 'MPLightFULL', 'FMA2CFull', 'FMA2CVAL'],
                        help="Choose the RL-based or rule-based agent to run.")

    parser.add_argument("--map", type=str, default='grid4x4',
                        choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                                 'cologne1', 'cologne3', 'cologne8'],
                        help="Specify the traffic network map.")

    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--eps", type=int, default=100, help="Number of episodes per trial.")
    parser.add_argument("--procs", type=int, default=1, help="Number of parallel processes to use.")

    parser.add_argument("--pwd", type=str, default=os.path.dirname(__file__), help="Project working directory.")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'results' + os.sep),
                        help="Directory to save logs and results.")

    parser.add_argument("--gui", type=bool, default=False, help="Whether to enable SUMO GUI.")
    parser.add_argument("--libsumo", type=bool, default=True, help="Use libsumo instead of traci.")
    parser.add_argument("--tr", type=int, default=0, help="Trial number (important when using libsumo).")

    parser.add_argument("--seps", type=int, default=0, help="Episode to start from when resuming training.")
    parser.add_argument("--load", type=bool, default=False, help="Whether to load a saved model.")

    parser.add_argument("--strategy", type=int, default=2, choices=[1, 2, 3],
                        help="Training strategy: 1 = base, 2 = incident, "
                             "3 = curriculum (planned, not yet implemented).")
    parser.add_argument("--repeat", type=int, default=0,
                        help="How many episodes to repeat incidents from training in testing.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for the agent. Defaults to the per-network/per-agent "
                             "value in TREX_comp/config/hyperparams.yaml (Appendix B) when omitted.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed, applied to Python's random, numpy, torch, "
                             "and SUMO (each episode gets seed+episode_index passed to SUMO's "
                             "--seed, instead of --random, for reproducibility). The paper's "
                             "results are averaged over 5 seeds -- run this flag with 5 "
                             "different values (e.g. 0-4) and average externally to reproduce "
                             "that. Pass --no-seed-sumo to fall back to SUMO's own --random "
                             "instead (Python/numpy/torch are still seeded either way).")
    parser.add_argument("--no-seed-sumo", action="store_true",
                        help="Launch SUMO with --random instead of a derived --seed "
                             "(pre-audit behavior). Python/numpy/torch are still seeded by --seed.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to a nonempty value to enable libsumo.")

    if args.strategy == 3:
        raise NotImplementedError(
            "--strategy 3 (curriculum) is planned but not yet implemented. "
            "Use --strategy 1 (base) or --strategy 2 (incident)."
        )

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
    mdp_config = mdp_configs.get(args.agent, {}).get(args.map)
    agt_config = agent_configs[args.agent]
    map_config = map_configs[args.map]

    if mdp_config:
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

    # --strategy is the single incidents on/off toggle (1 = base/off, 2 = incident/on;
    # 3 is rejected above before this point is reached). Both modes construct the same
    # TrexEnv class -- see trex_env.py.
    incident_config = DEFAULT_INCIDENTS if args.strategy == 2 else NO_INCIDENTS
    env = TrexEnv(
        run_name=f"{agt_config['agent'].__name__}-tr{trial}",
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
        incident_config=incident_config,
        sumo_seed=None if args.no_seed_sumo else args.seed,
    )

    # === Agent Setup ===
    alg = agt_config['agent']
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    agt_config.update({
        'episodes': int(args.eps * 0.8),
        'steps': int(args.eps * 0.8) * num_steps_eps,
        'log_dir': os.path.join(args.log_dir, env.connection_name),
        'num_lights': len(env.all_ts_ids),
        'save_freq': 50 if alg.__name__ in {'IPPO', 'FMA2C'} else 10,
        'load': args.load
    })

    obs_act = {
        key: [env.obs_shape[key], len(env.phases.get(key, []))]
        for key in env.obs_shape
    }

    if alg.__name__ in {'MPLight', 'IDQN'}:
        lr = resolve_learning_rate(alg.__name__, args.map, override=args.lr)
        agent = alg(agt_config, obs_act, args.map, trial, lr=lr) if lr is not None else \
                alg(agt_config, obs_act, args.map, trial)
    else:
        agent = alg(agt_config, obs_act, args.map, trial)

    # === Training or Testing ===
    try:
        if args.strategy == 1:
            run_base_scenario(env, agent, args, agt_config)
        else:
            run_incident_scenario(env, agent, args, agt_config, alg)
    finally:
        env.close()


# === Helper Functions ===

def run_base_scenario(env, agent, args, agt_config):
    if agt_config['load']:
        logger.info('Testing under base condition...')
        for _ in range(args.seps, args.eps):
            run_episode(env, agent)
    else:
        logger.info('Training under base condition...')
        for _ in range(args.eps):
            run_episode(env, agent)


def run_incident_scenario(env, agent, args, agt_config, alg):
    map_id = args.map
    agent_name = alg.__name__
    seed_file_1 = f"{agent_name}{map_id}-seed_ic1.txt"
    seed_file_2 = f"{agent_name}{map_id}-seed_ic2.txt"

    if agt_config['load']:
        logger.info('Testing under incident condition...')
        if args.repeat > 0 and os.path.exists(seed_file_1) and os.path.exists(seed_file_2):
            last_seeds_ic1 = load_seeds(seed_file_1, args.repeat)
            last_seeds_ic2 = load_seeds(seed_file_2, args.repeat)
            logger.info(f"Loaded seeds from files: {list(last_seeds_ic1)}, {list(last_seeds_ic2)}")

            for _ in range(args.seps, args.eps):
                seed_ic1, seed_ic2 = last_seeds_ic1.popleft(), last_seeds_ic2.popleft()
                obs = env.reset(pre_seed=[seed_ic1, seed_ic2])
                run_episode(env, agent, obs)
        else:
            logger.info('Testing without predefined incident seeds...')
            for _ in range(args.seps, args.eps):
                run_episode(env, agent)
    else:
        logger.info('Training under incident condition...')
        if args.repeat > 0:
            logger.info('Training and saving last incident seeds...')
            last_seed_ic1 = deque(maxlen=args.repeat)
            last_seed_ic2 = deque(maxlen=args.repeat)

            for ep in range(args.eps):
                obs = env.reset()
                if ep >= args.eps - args.repeat:
                    last_seed_ic1.append(env.seed_ic1)
                    last_seed_ic2.append(env.seed_ic2)
                run_episode(env, agent, obs)

            save_seeds(seed_file_1, last_seed_ic1)
            save_seeds(seed_file_2, last_seed_ic2)
        else:
            logger.info('Training without saving incident seeds...')
            for _ in range(args.eps):
                run_episode(env, agent)


def run_episode(env, agent, obs=None):
    if obs is None:
        obs = env.reset()
    done = False
    while not done:
        act = agent.act(obs)
        obs, rew, done, info = env.step(act)
        agent.observe(obs, rew, done, info)


def load_seeds(filename, max_len):
    with open(filename, "r") as f:
        return deque([int(line.strip()) for line in f.readlines()], maxlen=max_len)


def save_seeds(filename, seeds):
    with open(filename, "w") as f:
        for seed in seeds:
            f.write(f"{seed}\n")
    logger.info(f"Saved seeds to {filename}: {list(seeds)}")


if __name__ == "__main__":
    main()
