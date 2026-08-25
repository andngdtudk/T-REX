"""Unified SUMO/TraCI Gym environment for T-REX.

Consolidates what were, before this refactor, two separate and drifting
implementations (base_env.py::BaseEnv, incident_env.py::IncidentEnv) into a
single class. Both are now thin backward-compatible wrappers around this
module -- see their docstrings.

The RL-interaction plumbing (Gym env shape, phase/signal handling, TraCI
lifecycle) is adapted from the original: https://github.com/Pi-Star-Lab/RESCO
"""
import os
import logging

import gym
import numpy as np
import sumolib
import traci

from traffic_signal import Signal
from T_REX import Initializer, Deployment
from TREX_comp.config.incident_config import IncidentConfig

logger = logging.getLogger(__name__)


class TrexEnv(gym.Env):
    """SUMO-based traffic-signal-control Gym environment, with incidents optional.

    Incident behavior is controlled by a single parameter, `incident_config`:

    - `incident_config=None` (default): incidents disabled. The
      `Initializer`/`Deployment` subsystem (T_REX.py) is never constructed or
      invoked for the whole lifetime of the environment -- not run and
      discarded, not run with a zero incident count, literally never
      touched. This matters for reproducibility: `Initializer` draws from
      (and, once random, reseeds) the global numpy RNG, and if that ran even
      to decide "no incident this episode," it would consume RNG draws a
      true incident-free run wouldn't, silently shifting the random
      sequence seen downstream by route choice, demand generation, and RL
      exploration. See `tests/test_env_unification.py` for the regression
      test verifying this against the pre-refactor `BaseEnv`.
    - `incident_config=IncidentConfig(...)`: incidents enabled with the
      given parameters (`TREX_comp/config/incident_config.py`).

    A handful of other behaviors differed between the old `BaseEnv`/
    `IncidentEnv` beyond the incident subsystem itself -- whether the
    network's `.add.xml` (vType definitions) is loaded, and SUMO's global
    `--time-to-teleport` flag. These are preserved exactly per mode below
    (see `_seed_and_teleport_args`/`_build_sumo_command`) rather than
    converged, so that toggling `incident_config` doesn't silently change
    anything else about the simulation. One genuine bug found while
    diffing the two originals *was* fixed here (not preserved): `BaseEnv`
    built route-based `-r` paths as `self.route + '_N.rou.xml'` (a flat
    file-prefix convention) while `IncidentEnv` and `main.py`'s own
    decompress-check both used `os.path.join(self.route, ...)` (a
    subdirectory convention) -- these disagreed, and `BaseEnv` was
    consequently broken on `grid4x4`/`arterial4x4` (confirmed live: it
    raised `TraCIException: route file ... is not accessible`) whenever the
    network was decompressed the way `main.py`/the README instruct. See
    `AUDIT_REPORT.md` for the details; this refactor fixes it for both
    modes rather than reproducing the bug in the "incidents off" path.
    """

    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False,
                 end_time=3600, step_length=10, yellow_length=4, step_ratio=1,
                 max_distance=200, lights=(), log_dir='/', libsumo=False, warmup=0,
                 gymma=False, run=0, incident_config: IncidentConfig = None, sumo_seed=None):

        self.incident_config = incident_config
        self.enable_incidents = incident_config is not None

        # sumo_seed: base seed passed to SUMO's --seed (offset by episode number each
        # reset()) for reproducible vehicle-level stochasticity. None launches SUMO
        # with --random (unseeded) instead.
        self.sumo_seed = sumo_seed

        self.run = run
        self.map_name = map_name
        self.net = net
        self.route = route
        self.gui = gui
        self.log_dir = log_dir
        self.libsumo = libsumo
        self.gymma = gymma
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.max_distance = max_distance
        self.warmup = warmup
        self.start_time = 1
        self.sim_step = self.start_time
        self.sim_time = self.start_time
        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.step_ratio = step_ratio
        self.metrics = []
        self.wait_metric = {}
        self.sumo_cmd = None
        self.signal_ids = []
        self.incidents = []

        self.connection_name = f"{run_name}-{map_name}-{state_fn.__name__}-{reward_fn.__name__}"
        # Only resolved/used when incidents are enabled -- see _build_sumo_command.
        self.additional = self._find_additional_file() if self.enable_incidents else None
        self.scenario_folder = self._find_scenario_folder()

        logger.info(
            f"Initializing TrexEnv ({'incidents' if self.enable_incidents else 'base'}): "
            f"{self.connection_name}"
        )

        self.sumo = self._initialize_sumo()
        self._initialize_signals(lights)
        if self.enable_incidents:
            self._initialize_incidents(pre_seed=None)
        self._finalize_setup(run_name, lights)

        logger.info(f"Environment {self.connection_name} initialized successfully.")

    def _find_additional_file(self):
        if self.net.endswith('.sumocfg'):
            return self.net.replace('.sumocfg', '.add.xml')
        if self.net.endswith('.net.xml'):
            return self.net.replace('.net.xml', '.add.xml')
        return None

    def _find_scenario_folder(self):
        if self.route:
            return self.net
        if self.net.endswith('.sumocfg'):
            return self.net.replace('.sumocfg', '.net.xml')
        return self.net

    def _route_rou_xml(self, run_number):
        """Path to the route file for a given run/episode number.

        Subdirectory convention (os.path.join), matching main.py's own
        decompress-check and what the README instructs -- see the class
        docstring for why this differs from the pre-refactor BaseEnv.
        """
        return os.path.join(self.route, f"{self.map_name}_{run_number}.rou.xml")

    def _build_sumo_command(self):
        """SUMO command for the short-lived phase-detection connection in __init__."""
        if self.route:
            cmd = [sumolib.checkBinary('sumo'), '-n', self.net, '-r', self._route_rou_xml(1)]
        else:
            cmd = [sumolib.checkBinary('sumo'), '-c', self.net]
        if self.enable_incidents:
            cmd += ['-a', self.additional]
        cmd += ['--no-warnings', 'True']
        return cmd

    def _seed_and_teleport_args(self):
        """SUMO CLI args controlling RNG seeding and the teleport timeout.

        --seed makes the run reproducible; --random (the pre-audit default)
        does not. The per-episode offset (self.run) means each episode
        within one env instance still gets a distinct seed.

        Teleport handling differs by mode, preserved from the pre-refactor
        originals: with incidents disabled, SUMO's global teleport timeout
        is unconditionally turned off (--time-to-teleport -1), matching the
        old BaseEnv. With incidents enabled, the global timeout is left at
        SUMO's default and only vehicles genuinely queued behind an active
        incident are exempted per-vehicle (see
        T_REX.py::Deployment.manage_incident_queue_teleport_exemption),
        matching the old IncidentEnv.
        """
        args = ['--random'] if self.sumo_seed is None else ['--seed', str(self.sumo_seed + self.run)]
        if not self.enable_incidents:
            args += ['--time-to-teleport', '-1']
        return args

    def _initialize_sumo(self):
        sumo_cmd = self._build_sumo_command()
        if self.libsumo:
            traci.start(sumo_cmd)
            return traci
        else:
            traci.start(sumo_cmd, label=self.connection_name)
            return traci.getConnection(self.connection_name)

    def _initialize_signals(self, lights):
        self.signal_ids = self.sumo.trafficlight.getIDList()
        logger.info(f"Detected {len(self.signal_ids)} traffic lights: {self.signal_ids}")

        self.phases = {
            light_id: [
                p for p in self.sumo.trafficlight.getAllProgramLogics(light_id)[0].getPhases()
                if "y" not in p.state.lower() and "g" in p.state.lower()
            ]
            for light_id in self.signal_ids
        }

        self.all_ts_ids = lights if lights else self.signal_ids
        self.ts_starter = len(self.all_ts_ids)
        self.signals = {}
        self.obs_shape = {}
        self.observation_space = []
        self.action_space = []

        for ts in self.all_ts_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts])
        for ts in self.all_ts_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)

        observations = self.state_fn(self.signals)
        self.ts_order = []
        for ts, obs in observations.items():
            self.obs_shape[ts] = obs.shape
            self.observation_space.append(gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape))
            self.ts_order.append(ts)
            if ts in {'bot_left_mgr', 'bot_right_mgr', 'top_left_mgr', 'top_right_mgr', 'top_mgr', 'bot_mgr'}:
                self.action_space.append(4)
            else:
                self.action_space.append(gym.spaces.Discrete(len(self.phases[ts])))

        self.n_agents = len(self.all_ts_ids)

    def _finalize_setup(self, run_name, lights):
        if not self.libsumo:
            traci.switch(self.connection_name)
        traci.close()

        self.connection_name = f"{run_name}-{self.map_name}-{len(lights)}-{self.state_fn.__name__}-{self.reward_fn.__name__}"
        self.sumo_cmd = None

        full_log_dir = os.path.join(self.log_dir, self.connection_name)
        os.makedirs(full_log_dir, exist_ok=True)

        logger.info(f"Connection ID: {self.connection_name}")

    def step_sim(self):
        """Advance the SUMO simulation by step_ratio steps.

        When incidents are disabled, this loop body is *only*
        `self.sumo.simulationStep()` -- the incident subsystem is not
        consulted at all, per the class docstring.
        """
        for _ in range(self.step_ratio):
            if self.enable_incidents:
                for incident_info, incident in self.incidents:
                    if incident_info.is_incident:
                        incident.sim_incident(self.sim_step, reroute=True)
            self.sumo.simulationStep()
            self.sim_step += 1
            self.sim_time += 1

    def reset(self, pre_seed=(None, None)):
        """Reset the simulation environment."""
        self.sim_step = 1
        self.sim_time = 1

        if self.run != 0:
            if not self.libsumo:
                traci.switch(self.connection_name)
            try:
                traci.close()
            finally:
                self.save_metrics()

        self.metrics.clear()
        self.run += 1

        self.sumo_cmd = [sumolib.checkBinary('sumo-gui') if self.gui else sumolib.checkBinary('sumo')]
        if self.gui:
            self.sumo_cmd.append('--start')

        if self.route:
            self.sumo_cmd += ['-n', self.net, '-r', self._route_rou_xml(self.run)]
        else:
            self.sumo_cmd += ['-c', self.net]

        if self.enable_incidents:
            self.sumo_cmd += ['--additional-files', self.additional]
        self.sumo_cmd += self._seed_and_teleport_args()
        self.sumo_cmd += [
            '--tripinfo-output', os.path.join(self.log_dir, self.connection_name, f'tripinfo_{self.run}.xml'),
            '--tripinfo-output.write-unfinished',
            '--no-step-log', 'True',
            '--no-warnings', 'True',
        ]

        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        try:
            if self.enable_incidents:
                self._initialize_incidents(pre_seed)
            else:
                self.incidents = []

            for _ in range(self.warmup):
                self.step_sim()

            if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids):
                self.ts_starter += 1
            self.signal_ids = self.all_ts_ids[:self.ts_starter]

            for ts in self.signal_ids:
                self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts])
                self.wait_metric[ts] = 0.0
            for ts in self.signal_ids:
                self.signals[ts].signals = self.signals
                self.signals[ts].observe(self.step_length, self.max_distance)

            observations = self.state_fn(self.signals)
        except Exception:
            # Episode setup failed after SUMO was already (re)started -- close
            # the just-opened connection so it isn't leaked, then re-raise.
            if not self.libsumo:
                traci.switch(self.connection_name)
            traci.close()
            raise

        if self.gymma:
            return [observations[ts] for ts in self.ts_order]
        return observations

    def _initialize_incidents(self, pre_seed):
        """Initialize incidents for the current episode. Only called when enabled."""
        self.incidents = []

        for i in range(self.incident_config.level):
            pre = pre_seed[i] if isinstance(pre_seed, (list, tuple)) else None
            incident_info = Initializer(
                map_name=self.map_name,
                run_num=0,
                scenario_folder=self.scenario_folder,
                warm_up_time=self.warmup,
                is_random=self.incident_config.is_random,
                level=self.incident_config.level,
                pre_seed=pre,
            )
            if incident_info.is_random:
                incident_info.random()

            if incident_info.is_incident:
                incident = Deployment(incident_info)
                incident.traci_init()
                self.incidents.append((incident_info, incident))

            if i == 0:
                self.seed_ic1 = incident_info.random_seed
            elif i == 1:
                self.seed_ic2 = incident_info.random_seed

    def step(self, act):
        """Perform an environment step given an action."""
        if self.gymma:
            act = {ts: act[i] for i, ts in enumerate(self.ts_order)}

        for signal_id in act:
            self.signals[signal_id].prep_phase(act[signal_id])

        for _ in range(self.yellow_length):
            self.step_sim()

        for signal_id in self.signal_ids:
            self.signals[signal_id].set_phase()

        for _ in range(self.step_length - self.yellow_length):
            self.step_sim()

        for signal_id in self.signal_ids:
            self.signals[signal_id].observe(self.step_length, self.max_distance)

        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals)

        self.calc_metrics(rewards)

        done = self.sumo.simulation.getTime() >= self.end_time

        if self.gymma:
            obss = [observations[ts] for ts in self.ts_order]
            rww = [rewards[ts] for ts in self.ts_order]
            return obss, rww, [done], {'eps': self.run}
        return observations, rewards, done, {'eps': self.run}

    def calc_metrics(self, rewards):
        """Collect performance metrics."""
        queue_lengths = {}
        max_queues = {}
        for signal_id, signal in self.signals.items():
            queue_length = sum(signal.full_observation[lane]['queue'] for lane in signal.lanes)
            max_queue = max(signal.full_observation[lane]['queue'] for lane in signal.lanes)
            queue_lengths[signal_id] = queue_length
            max_queues[signal_id] = max_queue

        self.metrics.append({
            'step': self.sumo.simulation.getTime(),
            'reward': rewards,
            'max_queues': max_queues,
            'queue_lengths': queue_lengths,
        })

    def save_metrics(self):
        """Save collected metrics to CSV."""
        metrics_path = os.path.join(self.log_dir, self.connection_name, f'metrics_{self.run}.csv')
        logger.info(f"Saving metrics to {metrics_path}")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w+') as output_file:
            for line in self.metrics:
                csv_line = ', '.join(str(line[metric]) for metric in ['step', 'reward', 'max_queues', 'queue_lengths'])
                output_file.write(csv_line + '\n')

    def render(self, mode='human'):
        pass

    def close(self):
        """Properly close SUMO simulation."""
        if not self.libsumo:
            traci.switch(self.connection_name)
        try:
            traci.close()
        finally:
            self.save_metrics()
