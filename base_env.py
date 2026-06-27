
# Adapted from original: https://github.com/Pi-Star-Lab/RESCO

import os
import numpy as np
import traci
import sumolib
import gym
import traceback
from traffic_signal import Signal, ensure_map_signal_control_config, export_map_signal_config
from simulation_summary import (
    initialize_runtime_counters,
    update_runtime_counters,
    finalize_runtime_state,
    print_grouped_mode_summary,
)

class BaseEnv(gym.Env):
    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False, end_time=3600,
                 step_length=10, yellow_length=4, step_ratio=1, max_distance=200, lights=(), log_dir='/', libsumo=False,
                 warmup=0, gymma=False, run=0, level=None, max_green_hold_steps=12):
        self.libsumo = libsumo
        self.gymma = gymma  # gymma expects sequential list of states/rewards instead of dict
        print(map_name, net, state_fn.__name__, reward_fn.__name__)
        self.log_dir = log_dir
        self.net = net
        self.route = route
        self.gui = gui
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.max_distance = max_distance
        self.warmup = warmup
        self.max_green_hold_steps = max_green_hold_steps

        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.step_ratio = step_ratio
        self.connection_name = run_name + '-' + map_name + '---' + state_fn.__name__ + '-' + reward_fn.__name__
        self.map_name = map_name
        try:
            self.force_jupedsim = "kbh" in str(map_name)   # Force jupedsim for kbh maps
        except:
            self.force_jupedsim = False
            print("Error checking map name for jupedsim, defaulting to no forced jupedsim. Error was:")
            traceback.print_exc()

        disable_jupedsim = os.getenv("TREX_DISABLE_JUPEDSIM", "").strip().lower() in {"1", "true", "yes", "on"}
        if disable_jupedsim:
            self.force_jupedsim = False

        # Run some steps in the simulation with default light configurations to detect phases
        if self.route is not None:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-n', net, '-r', self.route + '_1.rou.xml', '--no-warnings', 'True', '--duration-log.statistics', 'False']
        else:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True', '--duration-log.statistics', 'False']
        if self.libsumo:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label = self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)
        self.signal_ids = self.sumo.trafficlight.getIDList()
        print("lights", len(self.signal_ids), self.signal_ids)

        # this should work on all SUMO versions
        self.phases = {
            lightID: [
                p
                for p in self.sumo.trafficlight.getAllProgramLogics(lightID)[0].getPhases()
                if "y" not in p.state and "g" in p.state.lower()
            ]
            for lightID in self.signal_ids
        }

        ensure_map_signal_control_config(self.map_name, self.phases)


        self.signals = dict()

        self.all_ts_ids = lights if len(lights) > 0 else self.sumo.trafficlight.getIDList()
        self.ts_starter = len(self.all_ts_ids)
        self.signal_ids = []

        # Pull signal observation shapes
        self.obs_shape = dict()
        self.observation_space = list()
        self.action_space = list()
        for ts in self.all_ts_ids:
            self.signals[ts] = Signal(
                self.map_name,
                self.sumo,
                ts,
                self.yellow_length,
                self.phases[ts],
                max_green_hold_steps=self.max_green_hold_steps,
            )
        exported_file = export_map_signal_config(self.map_name)
        if exported_file is not None:
            print('Generated signal config file:', exported_file)
        for ts in self.all_ts_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)
        observations = self.state_fn(self.signals)
        self.ts_order = list()
        for ts in observations:
            # if ts == 'top_mgr' or ts == 'bot_mgr': continue     # Not a traffic signal
            o_shape = observations[ts].shape
            self.obs_shape[ts] = o_shape
            o_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=o_shape)
            self.ts_order.append(ts)
            self.observation_space.append(o_shape)
            # if ts == 'top_mgr' or ts == 'bot_mgr': continue
            if ts == 'bot_left_mgr' or ts == 'bot_right_mgr' or ts == 'top_left_mgr' or ts == 'top_right_mgr' or ts == 'top_mgr' or ts == 'bot_mgr':
                self.action_space.append(4)
            else:
                self.action_space.append(gym.spaces.Discrete(len(self.phases[ts])))

        self.n_agents = self.ts_starter

        self.run = run
        self.metrics = []
        self.wait_metric = dict()
        self.summary_counters = initialize_runtime_counters()

        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.connection_name = run_name + '-' + map_name + '-' + str(len(lights)) + '-' + state_fn.__name__ + '-' + reward_fn.__name__
        if not os.path.exists(log_dir+self.connection_name):
            os.makedirs(log_dir+self.connection_name)
        self.sumo_cmd = None
        print('Connection ID', self.connection_name)

    def step_sim(self):
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        for _ in range(self.step_ratio):
            self.sumo.simulationStep()
            update_runtime_counters(self.sumo, self.summary_counters)
        
    def reset(self):
        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset()
        if self.run != 0:
            if not self.libsumo: traci.switch(self.connection_name)
            self._finalize_current_run_summary()
            traci.close()
            print_grouped_mode_summary(self.log_dir, self.connection_name, self.run, self.summary_counters)
            self.save_metrics()
        self.metrics = []


        self.run += 1

        # Start a new simulation
        self.sumo_cmd = []
        if self.gui:
            self.sumo_cmd.append(sumolib.checkBinary('sumo-gui'))
            self.sumo_cmd.append('--start')
        else:
            self.sumo_cmd.append(sumolib.checkBinary('sumo'))
        if self.route is not None:
            self.sumo_cmd += ['-n', self.net, '-r', self.route + '_'+str(self.run)+'.rou.xml']
        else:
            self.sumo_cmd += ['-c', self.net]
        self.sumo_cmd += ['--random', '--time-to-teleport', '-1', '--tripinfo-output',
                          os.path.join(self.log_dir, self.connection_name, 'tripinfo_' + str(self.run) + '.xml'),
                          '--personinfo-output',
                          os.path.join(self.log_dir, self.connection_name, 'personinfo_' + str(self.run) + '.xml'),
                          '--tripinfo-output.write-unfinished',
                          '--duration-log.statistics', 'False',
                          '--no-step-log', 'True',
                          '--no-warnings', 'True']
        if self.force_jupedsim:
            self.sumo_cmd += ['--pedestrian.model', 'jupedsim']
        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)
        self.summary_counters = initialize_runtime_counters()

        for _ in range(self.warmup):
            self.step_sim()

        # 'Start' only signals set for control, rest run fixed controllers
        if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids): self.ts_starter += 1
        self.signal_ids = []
        for i in range(self.ts_starter):
            self.signal_ids.append(self.all_ts_ids[i])

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

        if self.gymma:
            states = self.state_fn(self.signals)
            rets = list()
            for ts in self.ts_order:
                rets.append(states[ts])
            return rets

        return self.state_fn(self.signals)

    def step(self, act):
        if self.gymma:
            dict_act = dict()
            for i, ts in enumerate(self.ts_order):
                dict_act[ts] = act[i]
            act = dict_act

        # Send actions to their signals
        for signal in self.signals:
            self.signals[signal].prep_phase(act[signal])

        for step in range(self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].set_phase()
        for step in range(self.step_length - self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].observe(self.step_length, self.max_distance)

        # observe new state and reward
        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals, step)

        self.calc_metrics(rewards)

        done = self.sumo.simulation.getTime() >= self.end_time
        if self.gymma:
            obss, rww = list(), list()
            for ts in self.ts_order:
                obss.append(observations[ts])
                rww.append(rewards[ts])
            return obss, rww, [done], {'eps': self.run}

        # DEBUG
        # Diagnostic checks for "ped_pressure always 0" — run interactively or drop
        # these prints into your env loop for a handful of steps, NOT a standalone
        # script (it needs a live `signal` / `signals` from your running sim).

        #CHECK 0
        for signal_id, signal in self.signals.items():
            print(signal.sumo.trafficlight.getControlledLinks('J01'))

        # --- Check 1: is ped_crossings populated at all? ---
        for signal_id, signal in self.signals.items():
            n_crossings = len(getattr(signal, 'ped_crossings', {}))
            print(f"[check1] {signal_id}: ped_crossings = {n_crossings}")
            if n_crossings:
                # peek at one crossing's edge lists
                sample_dir, sample_crossing = next(iter(signal.ped_crossings.items()))
                print(f"          sample direction={sample_dir} "
                    f"in_edges={sample_crossing.get('in_edges')} "
                    f"out_edges={sample_crossing.get('out_edges')}")

        # --- Check 2: is _collect_ped_crossing_pressure ever being called? ---
        # Add a print/counter inside the method itself temporarily:
        #
        #   def _collect_ped_crossing_pressure(self):
        #       print(f"[check2] called for {self.id} at step {self.sumo.simulation.getTime()}")
        #       ...
        #
        # If this never prints during a training run, nothing calls it for this
        # signal/agent path — that's your bug, independent of SUMO pedestrian data.

        # --- Check 3: are there ANY persons in the simulation at all? ---
        try:
            all_person_ids = signal.sumo.person.getIDList()
            print(f"[check3] total persons in sim right now: {len(all_person_ids)}")
            if all_person_ids:
                pid = all_person_ids[0]
                print(f"          sample person {pid} on edge "
                    f"{signal.sumo.person.getRoadID(pid)}")
        except Exception as e:
            print(f"[check3] error querying persons: {e}")

        # --- Check 4: ped_detect_distance sane? ---
        print(f"[check4] ped_detect_distance = {getattr(signal, 'ped_detect_distance', None)}")

        # --- Check 5: manually replicate _collect_ped_crossing_pressure's inner loop
        #     for one crossing, with prints at each stage, to see where the count
        #     drops to zero ---
        for direction, crossing in list(signal.ped_crossings.items())[:1]:
            print(f"[check5] direction={direction}")
            for edge_id in crossing['in_edges']:
                try:
                    pids = signal.sumo.edge.getLastStepPersonIDs(edge_id)
                except Exception as e:
                    print(f"   in_edge {edge_id}: ERROR {e}")
                    continue
                print(f"   in_edge {edge_id}: getLastStepPersonIDs -> {pids}")
                for pid in pids:
                    within = signal._person_within_distance(pid, edge_id)
                    print(f"      person {pid}: within_distance={within}")

        return observations, rewards, done, {'eps': self.run}

    def calc_metrics(self, rewards):
        queue_lengths = dict()
        max_queues = dict()
        for signal_id in self.signals:
            signal = self.signals[signal_id]
            queue_length, max_queue = 0, 0
            for lane in signal.lanes:
                queue = signal.full_observation[lane]['queue']
                if queue > max_queue: max_queue = queue
                queue_length += queue
            queue_lengths[signal_id] = queue_length
            max_queues[signal_id] = max_queue
        self.metrics.append({
            'step': self.sumo.simulation.getTime(),
            'reward': rewards,
            'max_queues': max_queues,
            'queue_lengths': queue_lengths
        })

    def save_metrics(self):
        log = os.path.join(self.log_dir, self.connection_name+ os.sep + 'metrics_' + str(self.run) + '.csv')
        print('saving', log)
        with open(log, 'w+') as output_file:
            for line in self.metrics:
                csv_line = ''
                for metric in ['step', 'reward', 'max_queues', 'queue_lengths']:
                    csv_line = csv_line + str(line[metric]) + ', '
                output_file.write(csv_line + '\n')

    def render(self, mode='human'):
        pass

    def _finalize_current_run_summary(self):
        if self.run <= 0:
            return
        try:
            finalize_runtime_state(self.sumo, self.summary_counters)
        except Exception as exc:
            print(f'Could not finalize grouped mode summary state: {exc}')

    def close(self):
        if not self.libsumo: traci.switch(self.connection_name)
        self._finalize_current_run_summary()
        traci.close()
        print_grouped_mode_summary(self.log_dir, self.connection_name, self.run, self.summary_counters)
        self.save_metrics()