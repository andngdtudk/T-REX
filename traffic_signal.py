import os
import traci
import copy
from pathlib import Path
from pprint import pformat
from TREX_comp.config.signal_config import signal_configs


def create_yellows(phases, yellow_length):
    new_phases = copy.copy(phases)
    yellow_dict = {}    # current phase + next phase keyed to corresponding yellow phase index
    # Automatically create yellow phases, traci will report missing phases as it assumes execution by index order
    for i in range(0, len(phases)):
        for j in range(0, len(phases)):
            if i != j:
                need_yellow, yellow_str = False, ''
                for sig_idx in range(len(phases[i].state)):
                    if (phases[i].state[sig_idx] == 'G' or phases[i].state[sig_idx] == 'g') and (phases[j].state[sig_idx] == 'r' or phases[j].state[sig_idx] == 's'):
                        need_yellow = True
                        yellow_str += 'y'
                    else:
                        yellow_str += phases[i].state[sig_idx]
                if need_yellow:  # If a yellow is required
                    new_phases.append(traci.trafficlight.Phase(yellow_length, yellow_str))
                    yellow_dict[str(i) + '_' + str(j)] = len(new_phases) - 1  # The index of the yellow phase in SUMO
    return new_phases, yellow_dict


def ensure_map_signal_control_config(map_name, phases_by_signal):
    cfg = signal_configs.setdefault(map_name, {})
    if 'phase_pairs' in cfg and 'valid_acts' in cfg:
        return

    phase_pairs, valid_acts = infer_phase_pairs_and_valid_acts(phases_by_signal)
    cfg['phase_pairs'] = phase_pairs
    cfg['valid_acts'] = valid_acts

    # Emit copy-pasteable config snippet in the same style used in signal_config.py.
    print('GENERATED MAP CONTROL CONFIG')
    print("'" + map_name + "': {")
    print("'phase_pairs':" + str(phase_pairs) + ',')
    print("'valid_acts':" + str(valid_acts) + ',')
    print('},')


def export_map_signal_config(map_name):
    cfg = signal_configs.get(map_name)
    if cfg is None:
        return None

    if 'phase_pairs' not in cfg or 'valid_acts' not in cfg:
        return None

    out_dir = Path(__file__).resolve().parent / 'TREX_comp' / 'config' / 'generated'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{map_name}_signal_config.py'

    # Keep key order so output resembles the hand-written format in signal_config.py.
    map_block = {map_name: cfg}
    content = (
        '# Auto-generated signal config block\n'
        '# Paste the map entry into TREX_comp/config/signal_config.py if desired.\n\n'
        + pformat(map_block, sort_dicts=False, width=120)
    )
    out_file.write_text(content, encoding='utf-8')
    return str(out_file)


def infer_phase_pairs_and_valid_acts(phases_by_signal):
    phase_pairs = []
    valid_acts = {}

    for signal_id, phases in phases_by_signal.items():
        signal_valid = {}
        for local_phase_idx, phase in enumerate(phases):
            pair = infer_phase_pair_from_state(phase.state)
            # Keep one phase-pair entry per local phase to avoid collisions
            # when two phases map to the same inferred movement pair.
            phase_pairs.append(pair)
            pair_index = len(phase_pairs) - 1
            signal_valid[pair_index] = local_phase_idx
        valid_acts[signal_id] = signal_valid

    return phase_pairs, valid_acts


def infer_phase_pair_from_state(state):
    active_movements = []
    # Link-state strings are ordered by controlled-link index, but the number of
    # links per junction can vary widely. Bucket the full state into 12 bins so
    # every green bit contributes to one movement index used by MPLight/MAXPRESSURE.
    state_len = max(1, len(state))
    for idx, sig_state in enumerate(state):
        movement = min(11, int((idx * 12) / state_len))
        if sig_state == 'g' or sig_state == 'G':
            active_movements.append(movement)

    if len(active_movements) == 0:
        return [0, 1]

    # Rank movements by how many green links they control in this phase.
    counts = {}
    for movement in active_movements:
        counts[movement] = counts.get(movement, 0) + 1

    ranked = sorted(counts.keys(), key=lambda movement: (-counts[movement], movement))
    if len(ranked) == 1:
        return [ranked[0], ranked[0]]
    return [ranked[0], ranked[1]]


class Signal:
    def __init__(self, map_name, sumo, id, yellow_length, phases, max_green_hold_steps=12):
        self.map_name = map_name
        self.sumo = sumo
        self.id = id
        self.yellow_time = yellow_length
        self.next_phase = 0
        self.num_green_phases = len(phases)
        if max_green_hold_steps is None:
            self.max_green_hold_steps = None
        else:
            self.max_green_hold_steps = max(1, int(max_green_hold_steps))
        self._same_green_decisions = 0

        links = self.sumo.trafficlight.getControlledLinks(self.id)
        lanes = []

        #for i, link in enumerate(links):
        #    link = link[0]  # unpack so link[0] is inbound, link[1] outbound
        #    if link[0] not in lanes: lanes.append(link[0])
        #print(self.id, lanes)

        # Unique lanes
        self.lanes = []
        self.outbound_lanes = []

        reversed_directions = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

        # Group of lanes constituting a direction of traffic
        myconfig = signal_configs[map_name]
        if self.id in myconfig:
            self.lane_sets = myconfig[self.id]['lane_sets']
            self.lane_sets_outbound = self.lane_sets.copy()
            for key in self.lane_sets_outbound:     # Remove values from copy
                self.lane_sets_outbound[key] = []
            self.downstream = myconfig[self.id]['downstream']

            self.inbounds_fr_direction = dict()
            for direction in self.lane_sets:
                for lane in self.lane_sets[direction]:
                    inbound_to_direction = direction.split('-')[0]
                    inbound_fr_direction = reversed_directions[inbound_to_direction]
                    if inbound_fr_direction in self.inbounds_fr_direction:
                        dir_lanes = self.inbounds_fr_direction[inbound_fr_direction]
                        if lane not in dir_lanes:
                            dir_lanes.append(lane)
                    else:
                        self.inbounds_fr_direction[inbound_fr_direction] = [lane]
                    if lane not in self.lanes: self.lanes.append(lane)

            # Populate outbound lane information
            self.out_lane_to_signalid = dict()
            for direction in self.downstream:
                dwn_signal = self.downstream[direction]
                if dwn_signal is not None:  # A downstream intersection exists
                    dwn_lane_sets = myconfig[dwn_signal]['lane_sets']    # Get downstream signal's lanes
                    for key in dwn_lane_sets:   # Find all inbound lanes from upstream
                        if key.split('-')[0] == direction:    # Downstream direction matches
                            dwn_lane_set = dwn_lane_sets[key]
                            if dwn_lane_set is None: raise Exception('Invalid signal config')
                            for lane in dwn_lane_set:
                                if lane not in self.outbound_lanes: self.outbound_lanes.append(lane)
                                self.out_lane_to_signalid[lane] = dwn_signal
                                for selfkey in self.lane_sets:
                                    if selfkey.split('-')[1] == key.split('-')[0]:    # Out dir. matches dwnstrm in dir.
                                        self.lane_sets_outbound[selfkey] += dwn_lane_set
            for key in self.lane_sets_outbound:  # Remove duplicates
                self.lane_sets_outbound[key] = list(set(self.lane_sets_outbound[key]))
        else:
            self.generate_config()

        self.waiting_times = dict()     # SUMO's WaitingTime and AccumulatedWaiting are both wrong for multiple signals

        # Libsumo can become unstable when program logic is rewritten repeatedly
        # across resets. Keep native programs under libsumo and apply direct phase
        # switches only; preserve legacy yellow-program behavior for traci.
        use_libsumo_as_traci = bool(os.environ.get('LIBSUMO_AS_TRACI'))
        if use_libsumo_as_traci:
            self.phases = phases
            self.yellow_dict = {}
        else:
            self.phases, self.yellow_dict = create_yellows(phases, yellow_length)

            # logic = self.sumo.trafficlight.Logic(id, 0, 0, phases=self.phases) # not compatible with libsumo
            programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
            logic = programs[0]
            logic.type = 0
            logic.phases = self.phases
            self.sumo.trafficlight.setProgramLogic(self.id, logic)

        self.signals = None     # Used to allow signal sharing
        self.full_observation = None
        self.last_step_vehicles = None

    def _build_inbound_lane_to_signal_map(self):
        lane_to_signal = {}
        for signal_id in self.sumo.trafficlight.getIDList():
            for link_group in self.sumo.trafficlight.getControlledLinks(signal_id):
                if len(link_group) == 0:
                    continue
                for link in link_group:
                    inbound_lane = link[0]
                    if inbound_lane.startswith(':'):
                        continue
                    if inbound_lane not in lane_to_signal:
                        lane_to_signal[inbound_lane] = signal_id
                    break
        return lane_to_signal

    def _infer_downstream_signal(self, links, inbound_lanes, lane_to_signal):
        if inbound_lanes is None or len(inbound_lanes) == 0:
            return None

        for link_group in links:
            if len(link_group) == 0:
                continue
            for link in link_group:
                in_lane, out_lane = link[0], link[1]
                if in_lane.startswith(':') or out_lane.startswith(':'):
                    continue
                if in_lane not in inbound_lanes:
                    continue

                downstream_signal = lane_to_signal.get(out_lane)
                if downstream_signal is not None and downstream_signal != self.id:
                    return downstream_signal

        return None

    def generate_config(self):
        print('GENERATING CONFIG')
        # TODO raise Exception('Invalid signal config')
        index_to_movement = {0: 'S-W', 1: 'S-S', 2: 'S-E', 3: 'W-N', 4: 'W-W', 5: 'W-S', 6: 'N-E',
                             7: 'N-N', 8: 'N-W', 9: 'E-S', 10: 'E-E', 11: 'E-N'}
        reversed_directions = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}
        self.lane_sets = {}
        for idx, movement in index_to_movement.items():
            self.lane_sets[movement] = []
        self.lane_sets_outbound = {movement: [] for movement in index_to_movement.values()}
        self.downstream = {'N': None, 'E': None, 'S': None, 'W': None}

        links = self.sumo.trafficlight.getControlledLinks(self.id)
        for i, link_group in enumerate(links):
            if len(link_group) == 0:
                continue

            # Prefer external inbound lanes; internal connector lanes (':...')
            # should not be used as approach detectors.
            selected_link = None
            for link in link_group:
                if not link[0].startswith(':'):
                    selected_link = link
                    break
            if selected_link is None:
                continue

            if selected_link[0] not in self.lanes:
                self.lanes.append(selected_link[0])
            # Group of lanes constituting a direction of traffic
            # right, left, straight
            if i % 3 == 0:
                index = int(i/3)
                if index in index_to_movement:
                    self.lane_sets[index_to_movement[index]].append(selected_link[0])

        # Build inbound lanes grouped by the direction they come from.
        self.inbounds_fr_direction = {}
        for direction in self.lane_sets:
            for lane in self.lane_sets[direction]:
                inbound_to_direction = direction.split('-')[0]
                inbound_fr_direction = reversed_directions[inbound_to_direction]
                if inbound_fr_direction in self.inbounds_fr_direction:
                    dir_lanes = self.inbounds_fr_direction[inbound_fr_direction]
                    if lane not in dir_lanes:
                        dir_lanes.append(lane)
                else:
                    self.inbounds_fr_direction[inbound_fr_direction] = [lane]

        # Infer downstream intersections from outbound lanes of straight movements.
        lane_to_signal = self._build_inbound_lane_to_signal_map()
        self.downstream['N'] = self._infer_downstream_signal(links, self.lane_sets['S-S'], lane_to_signal)
        self.downstream['S'] = self._infer_downstream_signal(links, self.lane_sets['N-N'], lane_to_signal)
        self.downstream['E'] = self._infer_downstream_signal(links, self.lane_sets['W-W'], lane_to_signal)
        self.downstream['W'] = self._infer_downstream_signal(links, self.lane_sets['E-E'], lane_to_signal)

        print("'"+self.id+"'"+": {")
        print("'lane_sets':"+str(self.lane_sets)+',')
        print("'downstream':"+str(self.downstream)+'},')

        # Cache generated signal configuration in-memory so later resets can reuse it.
        map_config = signal_configs.get(self.map_name)
        if map_config is not None:
            map_config[self.id] = {
                'lane_sets': self.lane_sets,
                'downstream': self.downstream
            }

        # print(self.id)
        # print(self.sumo.trafficlight.getControlledLinks(self.id))
        # print(self.lanes)
        # print(self.outbound_lanes)
        # print(self.lane_sets_outbound)

    @property
    def phase(self):
        return self.sumo.trafficlight.getPhase(self.id)

    def prep_phase(self, new_phase):
        current_phase = int(self.phase)
        requested_phase = int(new_phase)

        # Guard against long-term starvation by forcing a phase change when
        # the same green phase has been repeatedly held for too long.
        if self.max_green_hold_steps is not None and self.num_green_phases > 1:
            if current_phase == requested_phase:
                self._same_green_decisions += 1
            else:
                self._same_green_decisions = 0

            if self._same_green_decisions >= self.max_green_hold_steps:
                requested_phase = (current_phase + 1) % self.num_green_phases
                self._same_green_decisions = 0

        if current_phase == requested_phase:
            self.next_phase = current_phase
        else:
            self.next_phase = requested_phase
            key = str(current_phase) + '_' + str(requested_phase)
            if key in self.yellow_dict:
                yel_idx = self.yellow_dict[key]
                self.sumo.trafficlight.setPhase(self.id, yel_idx)  # turns yellow

    def set_phase(self):
        self.sumo.trafficlight.setPhase(self.id, int(self.next_phase))

    def observe(self, step_length, distance):
        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'queue': 0, 'approach': 0, 'total_wait': 0, 'max_wait': 0}
            lane_vehicles = self.get_vehicles(lane, distance)
            for vehicle in lane_vehicles:
                all_vehicles.add(vehicle)
                # Update waiting time
                if vehicle in self.waiting_times:
                    self.waiting_times[vehicle] += step_length
                elif self.sumo.vehicle.getWaitingTime(vehicle) > 0:  # Vehicle stopped here, add it
                    self.waiting_times[vehicle] = self.sumo.vehicle.getWaitingTime(vehicle)

                vehicle_measures = dict()
                vehicle_measures['id'] = vehicle
                vehicle_measures['wait'] = self.waiting_times[vehicle] if vehicle in self.waiting_times else 0
                vehicle_measures['speed'] = self.sumo.vehicle.getSpeed(vehicle)
                vehicle_measures['acceleration'] = self.sumo.vehicle.getAcceleration(vehicle)
                vehicle_measures['position'] = self.sumo.vehicle.getLanePosition(vehicle)
                vehicle_measures['type'] = self.sumo.vehicle.getTypeID(vehicle)
                vehicles.append(vehicle_measures)
                if vehicle_measures['wait'] > 0:
                    lane_measures['total_wait'] = lane_measures['total_wait'] + vehicle_measures['wait']
                    lane_measures['queue'] = lane_measures['queue'] + 1
                    if vehicle_measures['wait'] > lane_measures['max_wait']:
                        lane_measures['max_wait'] = vehicle_measures['wait']
                else:
                    lane_measures['approach'] = lane_measures['approach'] + 1
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures

        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = all_vehicles.difference(self.last_step_vehicles)
            departs = self.last_step_vehicles.difference(all_vehicles)
            full_observation['departures'] = departs
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)

        self.last_step_vehicles = all_vehicles
        self.full_observation = full_observation

    # Remove undetectable vehicles from lane
    def get_vehicles(self, lane, max_distance):
        detectable = []
        for vehicle in self.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:  # Detectors have a max range
                    detectable.append(vehicle)
        return detectable
