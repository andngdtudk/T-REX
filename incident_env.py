import os
import gym
import numpy as np
import traci
import sumolib
from traffic_signal import Signal
from T_REX import Initializer, Deployment


class IncidentEnv(gym.Env):
    """Traffic signal control environment with incident simulation based on SUMO."""

    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False,
                 end_time=3600, step_length=10, yellow_length=4, step_ratio=1,
                 max_distance=300, lights=(), log_dir='/', libsumo=False, warmup=100, gymma=False, run=0, level=2,
                 max_green_hold_steps=12):

        # === Basic setup ===
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
        self.max_green_hold_steps = max_green_hold_steps
        self.warmup = warmup
        self.level = level
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

        self.connection_name = f"{run_name}-{map_name}-{state_fn.__name__}-{reward_fn.__name__}"
        self.force_jupedsim = map_name in {'kbh_full_multimodal', 'kbh_full_multimodal_mod'}
        self.additional = self._find_additional_file()
        self.scenario_folder = self._find_scenario_folder()

        print(f"Initializing IncidentEnv: {self.connection_name}")

        # === SUMO Initialization ===
        self.sumo = self._initialize_sumo()

        # === Traffic signals and observations ===
        self._initialize_signals(lights)

        # === Incident Management ===
        self._initialize_incidents(pre_seed=None)

        # === Final setup ===
        self._finalize_setup(run_name, lights)

        print(f"Environment {self.connection_name} initialized successfully.")

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

    def _build_sumo_command(self):
        if self.route:
            cmd = [
                sumolib.checkBinary('sumo'),
                '-n', self.net,
                '-r', os.path.join(self.route, f"{self.map_name}_1.rou.xml"),
                '-a', os.path.join(self.route, "vtypes.add.xml"),
                '--no-warnings', 'True'
            ]
            if self.force_jupedsim:
                cmd += ['--pedestrian.model', 'jupedsim']
            return cmd
        else:
            cmd = [
                sumolib.checkBinary('sumo'),
                '-c', self.net,
                '-a', self.additional,
                '--no-warnings', 'True'
            ]
            if self.force_jupedsim:
                cmd += ['--pedestrian.model', 'jupedsim']
            return cmd

    def _initialize_sumo(self):
        sumo_cmd = self._build_sumo_command()
        if self.libsumo:
            traci.start(sumo_cmd)
            return traci
        else:
            traci.start(sumo_cmd, label=self.connection_name)
            return traci.getConnection(self.connection_name)

    def _initialize_signals(self, lights):
        # Detect traffic lights
        self.signal_ids = self.sumo.trafficlight.getIDList()
        print(f"Detected {len(self.signal_ids)} traffic lights: {self.signal_ids}")

        # Detect valid phases (no yellow, at least one green)
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

        # Initialize signals and their observations
        for ts in self.all_ts_ids:
            self.signals[ts] = Signal(
                self.map_name,
                self.sumo,
                ts,
                self.yellow_length,
                self.phases[ts],
                max_green_hold_steps=self.max_green_hold_steps,
            )

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

        print(f"Connection ID: {self.connection_name}")
    
    def step_sim(self):
        """Advance the SUMO simulation by step_ratio steps, handle incidents if necessary."""
        for _ in range(self.step_ratio):
            for incident_info, incident in getattr(self, 'incidents', []):
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
            traci.close()
            self.save_metrics()

        self.metrics.clear()
        self.run += 1

        # Build new SUMO command
        self.sumo_cmd = [sumolib.checkBinary('sumo-gui') if self.gui else sumolib.checkBinary('sumo')]
        if self.gui:
            self.sumo_cmd.append('--start')

        if self.route:
            self.sumo_cmd += ['-n', self.net, '-r', os.path.join(self.route, f"{self.map_name}_{self.run}.rou.xml")]
        else:
            self.sumo_cmd += ['-c', self.net]

        self.sumo_cmd += [
            '--additional-files', self.additional,
            '--random',
            '--time-to-teleport', '-1',
            '--tripinfo-output', os.path.join(self.log_dir, self.connection_name, f'tripinfo_{self.run}.xml'),
            '--personinfo-output', os.path.join(self.log_dir, self.connection_name, f'personinfo_{self.run}.xml'),
            '--tripinfo-output.write-unfinished',
            '--no-step-log', 'True',
            '--no-warnings', 'True'
        ]
        if self.force_jupedsim:
            self.sumo_cmd += ['--pedestrian.model', 'jupedsim']

        # Restart SUMO
        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        # Reinitialize incidents
        self._initialize_incidents(pre_seed)

        # Warm-up simulation
        for _ in range(self.warmup):
            self.step_sim()

        # Dynamic agent control assignment
        if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids):
            self.ts_starter += 1

        self.signal_ids = self.all_ts_ids[:self.ts_starter]

        # Re-initialize controlled signals
        for ts in self.signal_ids:
            self.signals[ts] = Signal(
                self.map_name,
                self.sumo,
                ts,
                self.yellow_length,
                self.phases[ts],
                max_green_hold_steps=self.max_green_hold_steps,
            )
            self.wait_metric[ts] = 0.0

        for ts in self.signal_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)

        # Return observations
        observations = self.state_fn(self.signals)
        if self.gymma:
            return [observations[ts] for ts in self.ts_order]
        return observations

    def _initialize_incidents(self, pre_seed):
        """Initialize incidents for the current episode."""
        self.incidents = []

        for i in range(self.level):
            pre = pre_seed[i] if isinstance(pre_seed, (list, tuple)) else None
            incident_info = Initializer(
                map_name=self.map_name,
                run_num=0,
                scenario_folder=self.scenario_folder,
                warm_up_time=self.warmup,
                is_random=True,
                level=self.level,
                pre_seed=pre
            )
            if incident_info.is_random:
                incident_info.random()

            if incident_info.is_incident:
                incident = Deployment(incident_info)
                incident.traci_init()
                self.incidents.append((incident_info, incident))

            # Save seed for potential use
            if i == 0:
                self.seed_ic1 = incident_info.random_seed
            elif i == 1:
                self.seed_ic2 = incident_info.random_seed

    def step(self, act):
        """Perform an environment step given an action."""
        if self.gymma:
            act = {ts: act[i] for i, ts in enumerate(self.ts_order)}

        # Prepare and set phases
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

        # Gather observations and rewards
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
            'queue_lengths': queue_lengths
        })

    def save_metrics(self):
        """Save collected metrics to CSV."""
        metrics_path = os.path.join(self.log_dir, self.connection_name, f'metrics_{self.run}.csv')
        print(f"Saving metrics to {metrics_path}")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w+') as output_file:
            for line in self.metrics:
                csv_line = ', '.join(str(line[metric]) for metric in ['step', 'reward', 'max_queues', 'queue_lengths'])
                output_file.write(csv_line + '\n')

    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Properly close SUMO simulation."""
        if not self.libsumo:
            traci.switch(self.connection_name)
        traci.close()
        self.save_metrics()
