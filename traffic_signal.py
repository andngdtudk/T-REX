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


def ensure_map_signal_control_config(map_name, signals_by_id):
    """CHANGED SIGNATURE: now takes signals_by_id (dict[signal_id -> Signal],
    already constructed, with movement_index_map/movement_lanes populated)
    instead of phases_by_signal (dict[signal_id -> list[phase]]).
 
    Call this AFTER all Signal objects for this map have been constructed,
    but BEFORE any signal.observe() is called (observe() needs valid_acts
    to exist, via _collect_ped_crossing_pressure's lookup).
    """
    cfg = signal_configs.setdefault(map_name, {})
    if 'phase_pairs' in cfg and 'valid_acts' in cfg:
        return
 
    phase_pairs, valid_acts, num_movements = infer_phase_pairs_and_valid_acts(signals_by_id)
    cfg['phase_pairs'] = phase_pairs
    cfg['valid_acts'] = valid_acts
    cfg['num_movements'] = num_movements
 
    print('GENERATED MAP CONTROL CONFIG')
    print("'" + map_name + "': {")
    print("'phase_pairs':" + str(phase_pairs) + ',')
    print("'valid_acts':" + str(valid_acts) + ',')
    print("'num_movements':" + str(num_movements) + ',')
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


def infer_phase_pairs_and_valid_acts(signals_by_id):
    """CHANGED SIGNATURE: now takes signals_by_id (dict[signal_id ->
    Signal], already constructed) instead of phases_by_signal.
 
    Returns:
        phase_pairs: list[[movement_a, movement_b], ...]
        valid_acts: dict[signal_id -> dict[pair_index -> local_phase_idx]]
        num_movements: dict[signal_id -> int] -- NEW: since movement count
            can legitimately differ per signal (e.g. J01 has 11; another
            signal on the same map could have a different count), this is
            returned per-signal rather than as one global int. If your
            MPLight_MM/FRAP_MM construction currently expects a single
            global num_movements for the whole map (SharedAgent batching
            requirement), take max(num_movements.values()) and zero-pad
            shorter signals' state vectors up to that width -- same
            padding approach state_mplight_mm.py already uses for the
            pedestrian block, for the same SharedAgent reason.
    """
    phase_pairs = []
    valid_acts = {}
    num_movements = {}
 
    for signal_id, signal in signals_by_id.items():
        signal_valid = {}
        for local_phase_idx in range(signal.num_green_phases):
            pair = infer_phase_pair_from_movements(local_phase_idx, signal)
            phase_pairs.append(pair)
            pair_index = len(phase_pairs) - 1
            signal_valid[pair_index] = local_phase_idx
        valid_acts[signal_id] = signal_valid
        num_movements[signal_id] = len(signal.movement_index_map)
 
    return phase_pairs, valid_acts, num_movements

def infer_phase_pair_from_movements(local_phase_idx, signal):
    """Infer [movement_a, movement_b] for one local phase, using the REAL
    movement_index_map / movement_lanes already built on `signal` by
    Signal._build_movement_index_map() -- not a positional bucket.
 
    Always-green movements (signature == all 1s, i.e. green in every one
    of this signal's phases) are excluded from being selected as a paired
    movement: they carry no information about what's specifically being
    served by any ONE phase, and pairing a phase's real movement against
    one reproduces the exact dilution/starvation bug this was written to
    fix (confirmed on J01 phase 1, originally paired against an
    always-green satellite-junction lane via the old bucket scheme).
    """
    num_phases = signal.num_green_phases
    if not (0 <= local_phase_idx < num_phases):
        raise ValueError(
            f"local_phase_idx={local_phase_idx} out of range for "
            f"signal '{signal.id}' with {num_phases} phases"
        )

    all_green_sig = tuple([1] * num_phases)

    served = []
    for movement_idx, movement_key in signal.movement_index_map.items():
        signature, junction = movement_key  # unpack (signature, junction_prefix)
        if signature[local_phase_idx] == 1:
            n_lanes = len(signal.movement_lanes.get(movement_idx, []))
            served.append((movement_idx, signature, n_lanes))

    if len(served) == 0:
        raise ValueError(
            f"signal '{signal.id}' phase {local_phase_idx}: no movement is "
            f"green in this phase at all -- check movement_index_map."
        )

    non_always_green = [(m, n) for (m, sig, n) in served if sig != all_green_sig]
    ranked = sorted(non_always_green, key=lambda x: (-x[1], x[0]))

    if len(ranked) >= 2:
        return [ranked[0][0], ranked[1][0]]
    if len(ranked) == 1:
        return [ranked[0][0], ranked[0][0]]

    print(
        f"[WARNING] signal '{signal.id}' phase {local_phase_idx}: only "
        f"always-green movement(s) {[m for m, _, _ in served]} are green "
        f"here -- no non-always-green movement found for this phase. "
        f"Falling back to {served[0][0]}, but VERIFY this phase manually."
    )
    fallback = served[0][0]
    return [fallback, fallback]


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
        else:
            self.generate_config()

        # Populate outbound lane information (self.outbound_lanes,
        # self.out_lane_to_signalid, self.lane_sets_outbound) the SAME
        # way regardless of which branch above ran. The original code
        # only built this inside the "loaded from signal_configs" branch,
        # by matching myconfig[downstream_signal]['lane_sets'] direction
        # strings against self.lane_sets — which (a) silently left these
        # attributes UNSET for any signal that took the generate_config()
        # path (the bug this fixes: AttributeError on
        # out_lane_to_signalid), and (b) even where it "worked", depended
        # on the downstream signal already being present in
        # signal_configs with a 'lane_sets' key, which isn't guaranteed
        # if signals are constructed one at a time and the downstream
        # neighbor hasn't run generate_config() yet either.
        #
        # This replacement derives the same information directly from
        # SUMO's own controlled-link graph (getControlledLinks' in/out
        # lane pairing + _build_inbound_lane_to_signal_map, which queries
        # SUMO for every TLS regardless of Python-side construction
        # order) — fully general, no dependency on lane_sets direction
        # vocabulary or on other Signal objects already existing.
        self._build_outbound_lane_map()

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

        # Build stable phase → lane lookup tables.
        self._build_phase_lane_maps()

        # MPLight MM: derive per-movement lane groups from this signal's
        # own phase green/red structure (must run after _build_phase_lane_maps,
        # since it reuses self._flat_inbound_lanes).
        self._build_movement_index_map()

        # MPLight MM ped detection: pressure is read per-phase directly
        # from SUMO's own traci.trafficlight.getServedPersonCount() in
        # _collect_ped_crossing_pressure (called from observe()) — no
        # construction-time detection/setup needed, unlike three earlier
        # approaches that required building crossing/walking-area maps
        # here (see _collect_ped_crossing_pressure's docstring for why
        # those were replaced).
        self.ped_crossing_pressure = {}   # GLOBAL pair_idx -> float, set each observe()

        self.signals = None     # Used to allow signal sharing
        self.full_observation = None
        self.last_step_vehicles = None

    def _build_phase_lane_maps(self):
        """Build phase_lanes and phase_ped_lanes maps.

        phase_lanes[green_phase_idx]     → list of external vehicle inbound lanes
                                            that are actively served (state G or g).
        phase_ped_lanes[green_phase_idx] → list of pedestrian/crossing lanes
                                            that are actively served (state W or w).

        Only the first num_green_phases phases are processed; yellow phases
        appended by create_yellows are excluded: all lanes show 0 during yellows.

        Uses getControlledLinks as the stable positional index that aligns with
        phase.state characters — both sequences are ordered identically by SUMO.
        """
        raw_links = self.sumo.trafficlight.getControlledLinks(self.id)

        # Flatten: raw_links is a list of link-groups (one per state position).
        # Each group contains one or more (inbound, via, outbound) tuples.
        # We only need the inbound lane (index 0) per group.
        flat_inbound = []
        for group in raw_links:
            if len(group) == 0:
                # Phantom link — no lane to record, but position still counts.
                flat_inbound.append(None)
            else:
                # Take the first non-internal inbound lane in the group.
                chosen = None
                for link in group:
                    lane = link[0] if len(link) > 0 else None
                    if lane and not lane.startswith(':'):
                        chosen = lane
                        break
                # Fall back to any lane (even internal) so the position is filled.
                if chosen is None and group[0]:
                    chosen = group[0][0] if len(group[0]) > 0 else None
                flat_inbound.append(chosen)

        self._flat_inbound_lanes = flat_inbound  # reused by _build_movement_index_map

        self.phase_lanes = {}
        self.phase_ped_lanes = {}

        for phase_idx in range(self.num_green_phases):
            phase = self.phases[phase_idx]
            vehicle_lanes = []
            ped_lanes = []

            for pos, char in enumerate(phase.state):
                if pos >= len(flat_inbound):
                    break
                lane = flat_inbound[pos]
                if lane is None:
                    continue

                if char in ('G', 'g'):
                    # Vehicle or cyclist green — exclude internal connector lanes.
                    if not lane.startswith(':') and lane not in vehicle_lanes:
                        vehicle_lanes.append(lane)
                elif char in ('W', 'w'):
                    # Pedestrian/cyclist walk signal (crossing or sidewalk).
                    if lane not in ped_lanes:
                        ped_lanes.append(lane)

            self.phase_lanes[phase_idx] = vehicle_lanes
            self.phase_ped_lanes[phase_idx] = ped_lanes

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

    def _build_outbound_lane_map(self):
        """Populate self.outbound_lanes and self.out_lane_to_signalid
        directly from SUMO's controlled-link graph, independent of
        lane_sets direction vocabulary and independent of whether other
        Signal objects have finished constructing yet.

        Fixes: the original code only built these two attributes inside
        the "loaded from signal_configs" branch of __init__, by matching
        direction-string keys between self.lane_sets and the downstream
        signal's myconfig[...]['lane_sets'] entry. Any signal taking the
        generate_config() branch never got these attributes set at all —
        surfaced as `AttributeError: 'Signal' object has no attribute
        'out_lane_to_signalid'` the first time a state/reward function
        that reads it runs (this includes vanilla mplight()/pressure()
        too, not just mplight_mm() — they read the same attributes).

        General approach: for every controlled-link position of THIS
        signal, take its (in_lane, out_lane) pair from getControlledLinks
        directly. If out_lane is itself controlled by some OTHER traffic
        light (found via _build_inbound_lane_to_signal_map, which queries
        SUMO's full TLS list — not Python-side Signal objects), that
        out_lane is one of this signal's outbound_lanes, and that other
        TLS is its owner in out_lane_to_signalid. No dependency on
        lane_sets, no dependency on construction order.

        Also rebuilds self.lane_sets_outbound, grouping each outbound lane
        under the SAME direction key as the inbound lane sharing its
        controlled-link position (so existing code reading
        lane_sets_outbound[direction] still works) — this is best-effort
        bookkeeping for direction-keyed consumers; movement-indexed
        consumers (mplight_mm's state fn) should prefer
        self.movement_out_lanes instead, which doesn't go through this
        direction-string detour at all.
        """
        self.outbound_lanes = []
        self.out_lane_to_signalid = dict()
        self.lane_sets_outbound = {direction: [] for direction in getattr(self, 'lane_sets', {})}

        lane_to_signal = self._build_inbound_lane_to_signal_map()
        links = self.sumo.trafficlight.getControlledLinks(self.id)

        # Map each inbound lane to the lane_sets direction key it belongs
        # to, so outbound lanes sharing that position can be filed under
        # the same direction (best-effort; only meaningful if lane_sets
        # is direction-keyed, which it is for both __init__ branches).
        #
        # KNOWN LIMITATION, confirmed against real data: on irregular/
        # joined junctions (e.g. J01 on kbh_j1_442m), generate_config()'s
        # `index = int(i / 3)` bucketing can file the SAME lane under
        # TWO different direction keys (observed directly: J01's printed
        # lane_sets has lanes like '01.01#O0_1' appearing in both 'S-S'
        # and 'S-E'). When that happens here, whichever direction is
        # iterated last in this loop silently "wins" the tag — arbitrary,
        # not geometrically meaningful. This only affects the BEST-EFFORT
        # self.lane_sets_outbound bookkeeping below; it does NOT affect
        # mplight_mm's state function, which uses self.movement_lanes /
        # self.movement_out_lanes instead (derived from phase
        # green-signatures, not from this direction taxonomy, precisely
        # because the taxonomy breaks down on irregular junctions like
        # this one). If something else in your codebase still reads
        # lane_sets_outbound[direction] for J01-like signals, treat its
        # output as unreliable.
        inbound_lane_to_direction = {}
        for direction, lanes in getattr(self, 'lane_sets', {}).items():
            for lane in lanes:
                inbound_lane_to_direction[lane] = direction

        for link_group in links:
            if len(link_group) == 0:
                continue
            for link in link_group:
                if len(link) < 2:
                    continue
                in_lane, out_lane = link[0], link[1]
                if in_lane is None or out_lane is None:
                    continue
                if in_lane.startswith(':') or out_lane.startswith(':'):
                    continue

                dwn_signal = lane_to_signal.get(out_lane)
                if dwn_signal is None or dwn_signal == self.id:
                    continue  # not controlled by a downstream TLS, or loops back to self

                if out_lane not in self.outbound_lanes:
                    self.outbound_lanes.append(out_lane)
                self.out_lane_to_signalid[out_lane] = dwn_signal

                direction = inbound_lane_to_direction.get(in_lane)
                if direction is not None and out_lane not in self.lane_sets_outbound[direction]:
                    self.lane_sets_outbound[direction].append(out_lane)

    def _build_movement_index_map(self):
        """Derive movement_index -> controlled-link lanes, fully from this
        signal's own phase structure — no fixed direction taxonomy (no
        'S-W'/'N-N' categories, no positional bucket formula), so this
        generalizes to any intersection geometry, including joined or
        irregular junctions and secondary satellite junctions folded into
        the same TLS (e.g. a bike-crossing gate riding along on the main
        4 phases, as in this map's screenshot).
    
        THE RULE USED: a "movement" is a distinct (green/red signature,
        junction) pair across this signal's own phases. For each controlled-
        link position, build a tuple of which phases it's green ('G'/'g') in,
        AND note which physical junction the lane belongs to (the part of the
        lane ID before '#', e.g. '01.01' vs '01.02'). Two positions are only
        grouped into the same movement if they share BOTH the same green/red
        signature AND the same junction prefix.
    
        *** WHY THE JUNCTION PREFIX MATTERS ***
        An earlier version grouped purely by signature. On J01 this merged
        '01.01#N0_1' (main junction) and '01.02#S0_1' (satellite junction,
        the bike-crossing gate's companion lane) into a single "movement 8"
        just because both happen to be always-green across this signal's 4
        phases. They are physically unrelated approach lanes on different
        junctions, so collapsing them into one movement was wrong: it
        double-purposed phase_pairs entries that referenced movement 8 (e.g.
        phase_pairs[1] = [1, 8]) to mean two different physical locations at
        once, and diluted/confused the demand signal the network learned to
        associate with that phase pair, which produced a learned policy that
        systematically avoided selecting it (verified against J01 deployment:
        the agent essentially never served phase 1's real movement).
        Splitting by (signature, junction_prefix) keeps every other grouping
        decision from the original method unchanged, but stops semantically
        unrelated lanes on different junctions from ever being merged purely
        because their timing coincides.
    
        Positions green in NO real phase (unused/phantom links) are still
        excluded entirely, same as before — they're not a movement at all.
    
        Sets:
            self.movement_index_map: dict[int -> (signature, junction_prefix)]
                — movement_index -> the (green-phase signature, junction)
                pair defining it, e.g. ((1,0,0,1), '01.01') meaning "green in
                phase 0 and phase 3, on junction 01.01".
            self.movement_lanes: dict[int -> list[str]] — movement_index ->
                controlled-link INBOUND lanes sharing that (signature,
                junction) pair.
            self.movement_out_lanes: dict[int -> list[str]] — movement_index
                -> the corresponding OUTBOUND lanes, derived directly from
                getControlledLinks' (in_lane, out_lane) pairing.
    
        Ordering: movements are numbered by each (signature, junction) pair's
        FIRST occurrence position across the phase strings (phase 0 before
        phase 1, left to right within a phase). Always-green groups are no
        longer forced to the end as a single block — since they may now be
        split into multiple distinct movements (one per junction), each
        split piece is ordered by its own first-occurrence position like any
        other movement. This means movement indices may differ from a
        previous run that used pure-signature grouping; phase_pairs MUST be
        regenerated/re-verified against the new self.movement_index_map
        printed below, not assumed to still match the old indices.
    
        *** STILL REQUIRES YOUR VERIFICATION ***
        The junction-prefix split assumes junction membership is fully and
        correctly captured by the lane-ID prefix before '#' (true for J01's
        '01.01' vs '01.02' naming, per the map screenshot's main+satellite
        junction structure) — verify this convention holds for any other
        map you apply this to before trusting its movement counts.
        """
        flat_inbound = getattr(self, '_flat_inbound_lanes', None)
        if flat_inbound is None:
            raise RuntimeError(
                "_build_movement_index_map requires _build_phase_lane_maps "
                "to have run first (needs self._flat_inbound_lanes)."
            )
    
        raw_links = self.sumo.trafficlight.getControlledLinks(self.id)
        flat_outbound = []
        for group in raw_links:
            if len(group) == 0:
                flat_outbound.append(None)
                continue
            chosen = None
            for link in group:
                out_lane = link[1] if len(link) > 1 else None
                if out_lane and not out_lane.startswith(':'):
                    chosen = out_lane
                    break
            if chosen is None and group[0] and len(group[0]) > 1:
                chosen = group[0][1]
            flat_outbound.append(chosen)
    
        def junction_prefix(lane):
            """Lane ID prefix before '#', identifying the physical junction
            this lane belongs to, e.g. '01.01#N0_1' -> '01.01'. Falls back to
            the whole lane id if no '#' is present (shouldn't happen for real
            controlled-link lanes on this map convention, but avoids a crash
            if it does)."""
            return lane.split('#')[0] if '#' in lane else lane
    
        num_phases = self.num_green_phases
        group_to_lanes = {}
        group_to_out_lanes = {}
        group_first_pos = {}
    
        state_len = len(self.phases[0].state) if num_phases > 0 else 0
        for pos in range(state_len):
            lane = flat_inbound[pos] if pos < len(flat_inbound) else None
            if lane is None or lane.startswith(':'):
                continue  # phantom / internal connector — not a real approach lane
    
            signature = tuple(
                1 if (pos < len(self.phases[p].state) and self.phases[p].state[pos] in ('G', 'g')) else 0
                for p in range(num_phases)
            )
    
            if sum(signature) == 0:
                continue  # never green in any real phase — not a movement at all
    
            # Group by (signature, junction) instead of signature alone — see
            # docstring for why this matters (always-green merge bug on J01).
            group_key = (signature, junction_prefix(lane))
    
            if group_key not in group_to_lanes:
                group_to_lanes[group_key] = []
                group_to_out_lanes[group_key] = []
                group_first_pos[group_key] = pos
            if lane not in group_to_lanes[group_key]:
                group_to_lanes[group_key].append(lane)
    
            out_lane = flat_outbound[pos] if pos < len(flat_outbound) else None
            if out_lane and not out_lane.startswith(':') and out_lane not in group_to_out_lanes[group_key]:
                group_to_out_lanes[group_key].append(out_lane)
    
        # Order purely by first-occurrence position — no special-casing for
        # always-green groups anymore, since they may now be split across
        # multiple junctions and each split piece should sort independently.
        ordered_groups = sorted(group_to_lanes.keys(), key=lambda g: group_first_pos[g])
    
        self.movement_index_map = {i: g for i, g in enumerate(ordered_groups)}
        self.movement_lanes = {i: group_to_lanes[g] for i, g in enumerate(ordered_groups)}
        self.movement_out_lanes = {i: group_to_out_lanes[g] for i, g in enumerate(ordered_groups)}
    
        print(
            f"[{self.id}] derived {len(self.movement_index_map)} movements from "
            f"phase green-signatures split by junction: {self.movement_index_map}"
        )
        print(f"[{self.id}] movement -> inbound lanes: {self.movement_lanes}")
        print(f"[{self.id}] movement -> outbound lanes: {self.movement_out_lanes}")
    
        # Flag any always-green movements explicitly, same spirit as before,
        # but now per-junction rather than merged.
        all_green_sig = tuple([1] * num_phases) if num_phases > 0 else tuple()
        for i, (sig, junc) in self.movement_index_map.items():
            if sig == all_green_sig:
                print(
                    f"[{self.id}] *** movement {i} is an ALWAYS-GREEN movement on "
                    f"junction '{junc}' (green in every one of this signal's "
                    f"{num_phases} phases) — lanes: {self.movement_lanes[i]}. This "
                    f"is now isolated to lanes on junction '{junc}' only (no "
                    f"longer merged with always-green lanes on other junctions). "
                    f"VERIFY this is the movement set you expect before trusting "
                    f"any phase_pairs entries that reference index {i}."
                )
    
        print(
            f"[{self.id}] >>> movement indices may have CHANGED from a previous "
            f"pure-signature run. Regenerate/re-verify phase_pairs against this "
            f"printed mapping before training — do not assume old indices "
            f"(e.g. a previous 'movement 8') still refer to the same lanes."
        )

    def _collect_ped_crossing_pressure(self):
        """Per-phase-pair pedestrian pressure, using SUMO's own built-in
        traci.trafficlight.getServedPersonCount(tlsID, phaseIndex) —
        "returns the number of persons that would be served in the given
        phase" — instead of hand-rolled walking-area/crossing detection.

        REPLACES THREE EARLIER ATTEMPTS, all of which had real, confirmed
        problems:
          1. 'W'/'w' phase.state + cardinal-direction bucket — wrong,
             J01's phases never use 'W'/'w'.
          2. Forward/backward tracing through chained internal lanes to
             find real edges on each side of a crossing — broke down
             because J01's crossings/walking-areas form one connected
             ring, so every crossing ended up reporting identical
             pressure once the tracing went deep enough to succeed at
             all.
          3. Reading getLastStepPersonIDs() directly on the walking-area
             lane adjacent to each crossing (after fixing a lane-vs-edge
             id bug, and after fixing a single-sample-per-RL-decision
             timing gap with a per-substep polling accumulator) — STILL
             read 0.0 in a live run. Investigated via person.getRoadID()
             ground-truth tracing: a real pedestrian's reported position
             jumped directly between two NAMED edges (e.g. '01.01#s0' ->
             '01.01#N0') and never showed ANY internal walking-area or
             crossing lane at any sampled instant, despite confirmed
             dwell times of several seconds on those lanes by geometry
             (length / max pedestrian speed). This means
             getLastStepPersonIDs()/getRoadID() do not expose
             fine-grained walking-area/crossing transit for this
             pedestrian model the way the SUMO docs' general description
             of the striping model seemed to suggest — confirmed
             empirically, not assumed.

        getServedPersonCount sidesteps ALL of this: it's SUMO's own C++
        internal logic for exactly this question ("how many people are
        waiting to use a crossing that would be served by this phase"),
        using getNextEdge()-based intent filtering and walking
        forwards/backwards across the crossing — the same thing we were
        trying to reconstruct by hand from the Python API, but via
        internals that evidently see what our calls didn't. It is keyed
        directly by LOCAL SUMO PHASE INDEX, not by crossing/walking-area
        lane id at all — which also means none of the lane detection,
        edge conversion, ring-topology, or sub-step-polling machinery
        from the earlier attempts is needed anymore.

        Returns dict[pair_idx -> float], where pair_idx is the GLOBAL
        phase_pairs index (matching mplight_mm's state/reward functions),
        not a SUMO phase index — converted via signal_configs[map]
        ['valid_acts'][self.id], the same mapping phase_pairs/valid_acts
        already provide. Stored on self.ped_crossing_pressure AND
        self.phase_pair_ped_pressure (same dict, see note below on the
        attribute rename).

        NOTE ON NAMING: earlier versions exposed
        self.phase_pair_ped_crossings (pair_idx -> list of crossing ids)
        as a separate intermediate structure, consumed by
        state_mplight_mm.py's per-pair pedestrian block. That
        intermediate structure no longer exists — there ARE no crossing
        ids anymore, since getServedPersonCount works directly per phase.
        self.ped_crossing_pressure is now ALREADY keyed by global pair_idx
        (an int), not by crossing_id (a string like 'c0'). state_mplight_mm.py
        has been updated to match (see that file's comments).
        """
        pressure = {}
        myconfig = signal_configs.get(self.map_name, {})
        valid_acts = myconfig.get('valid_acts', {}).get(self.id, {})

        for pair_idx, local_phase_idx in valid_acts.items():
            try:
                count = self.sumo.trafficlight.getServedPersonCount(self.id, local_phase_idx)
            except Exception as e:
                print(f"[{self.id}] ERROR getServedPersonCount(phase={local_phase_idx}) "
                      f"for pair_idx {pair_idx}: {e}")
                count = 0
            pressure[pair_idx] = float(count)

        self.ped_crossing_pressure = pressure
        return pressure

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
            inbound_lanes = []
            for link in link_group:
                if link[0].startswith(':'):
                    continue
                if link[0] not in inbound_lanes:
                    inbound_lanes.append(link[0])
            if not inbound_lanes:
                continue

            for lane_id in inbound_lanes:
                if lane_id not in self.lanes:
                    self.lanes.append(lane_id)
            # Group of lanes constituting a direction of traffic
            # right, left, straight
            index = int(i / 3)
            if index in index_to_movement:
                movement = index_to_movement[index]
                for lane_id in inbound_lanes:
                    if lane_id not in self.lane_sets[movement]:
                        self.lane_sets[movement].append(lane_id)

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

    def _is_bike_vehicle(self, vehicle_id, vehicle_type):
        vehicle_class = self.sumo.vehicle.getVehicleClass(vehicle_id)
        if vehicle_class == 'bicycle':
            return True

        lowered_type = vehicle_type.lower()
        return 'bike' in lowered_type or 'bicycle' in lowered_type or 'cycle' in lowered_type

    def _get_controlled_edges(self):
        """Get the set of edges controlled by this signal so we can use them for pedestrian detection."""
        controlled_edges = set()
        links = self.sumo.trafficlight.getControlledLinks(self.id)
        for link_group in links:
            if len(link_group) == 0:
                continue
            for link in link_group:
                if len(link) < 2:
                    continue
                in_lane = link[0]
                out_lane = link[1]
                for lane_id in (in_lane, out_lane):
                    if lane_id is None or lane_id == '':
                        continue
                    try:
                        controlled_edges.add(self.sumo.lane.getEdgeID(lane_id))
                    except Exception as e:
                        print ("Error getting edge for lane {}: {}".format(lane_id, str(e)))
                        raise e
        return controlled_edges

    def _collect_pedestrian_measures(self, controlled_edges):
        """Collect pedestrian-related measures for the given set of controlled edges."""
        person_ids = set()
        for edge_id in controlled_edges:
            try:
                edge_person_ids = self.sumo.edge.getLastStepPersonIDs(edge_id)
            except Exception as e:
                print ("Error getting person IDs for edge {}: {}".format(edge_id, str(e)))
                raise e
            
            for person_id in edge_person_ids:
                person_ids.add(person_id)

        ped_waiting = 0
        ped_total_wait = 0.0
        ped_max_wait = 0.0
        ped_approaching_crossing = 0
        ped_leaving_intersection = 0

        for person_id in person_ids:
            try:
                waiting_time = self.sumo.person.getWaitingTime(person_id)
            except Exception as e:
                print ("Error getting waiting time for person {}: {}".format(person_id, str(e)))
                waiting_time = 0.0

            if waiting_time > 0:
                ped_waiting += 1
                ped_total_wait += waiting_time
                if waiting_time > ped_max_wait:
                    ped_max_wait = waiting_time

            try:
                next_edge = self.sumo.person.getNextEdge(person_id)
            except Exception:
                next_edge = None

            if next_edge in controlled_edges:
                ped_approaching_crossing += 1
            else:
                ped_leaving_intersection += 1

        return {
            'ped_ids': person_ids,
            'ped_count': len(person_ids),
            'ped_waiting': ped_waiting,
            'ped_total_wait': ped_total_wait,
            'ped_max_wait': ped_max_wait,
            'ped_approaching_crossing': ped_approaching_crossing,
            'ped_leaving_intersection': ped_leaving_intersection,
        }

    def observe(self, step_length, distance):
        full_observation = dict()
        all_vehicles = set()
        controlled_edges = self._get_controlled_edges()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {
                'queue': 0,
                'approach': 0,
                'total_wait': 0,
                'max_wait': 0,
                'bike_queue': 0,
                'bike_total_wait': 0,
                'bike_max_wait': 0,
            }
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
                vehicle_measures['is_bike'] = self._is_bike_vehicle(vehicle, vehicle_measures['type'])
                vehicles.append(vehicle_measures)
                if vehicle_measures['wait'] > 0:
                    lane_measures['total_wait'] = lane_measures['total_wait'] + vehicle_measures['wait']
                    lane_measures['queue'] = lane_measures['queue'] + 1
                    if vehicle_measures['wait'] > lane_measures['max_wait']:
                        lane_measures['max_wait'] = vehicle_measures['wait']

                    if vehicle_measures['is_bike']:
                        lane_measures['bike_total_wait'] = lane_measures['bike_total_wait'] + vehicle_measures['wait']
                        lane_measures['bike_queue'] = lane_measures['bike_queue'] + 1
                        if vehicle_measures['wait'] > lane_measures['bike_max_wait']:
                            lane_measures['bike_max_wait'] = vehicle_measures['wait']
                else:
                    lane_measures['approach'] = lane_measures['approach'] + 1
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures
        
        # Collect pedestrian measures now, as these are not lane based
        full_observation.update(self._collect_pedestrian_measures(controlled_edges))
        full_observation['ped_crossing_pressure'] = self._collect_ped_crossing_pressure()


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