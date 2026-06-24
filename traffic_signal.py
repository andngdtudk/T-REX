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

        # MPLight MM ped detection
        self.ped_detect_distance = float(
            signal_configs.get(map_name, {}).get('ped_detect_distance', 15.0)
        )
        self._build_ped_crossing_groups()
        self._build_phase_pair_ped_crossings()
        self.ped_crossing_pressure = {}   # crossing_id -> float, set each observe()

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

        WHY NOT signal.lane_sets: lane_sets always has up to 12 fixed
        turn x approach-direction keys from generate_config(), but real
        intersections vary in how many of those 12 are populated, and for
        irregular junctions phase_pairs' indexing often doesn't
        correspond to that scheme at all — confirmed directly against
        J01: all 12 lane_sets directions were populated (no empties to
        drop), yet phase_pairs only ever needs 9 movement indices. So
        lane_sets simply isn't the right abstraction here.

        THE RULE USED INSTEAD: a "movement" is a distinct green/red
        pattern across this signal's own phases. For each controlled-link
        position, build a tuple of which phases it's green ('G'/'g') in.
        Two positions sharing a tuple always turn green/red together, so
        FRAP can treat them as one movement. Positions green in NO real
        phase (unused/phantom links) are excluded — they're not a
        movement at all. Each remaining distinct signature becomes one
        movement index.

        *** ALWAYS-GREEN POSITIONS (signature = green in every phase) ***
        An earlier version of this method excluded these outright as
        "uncontested". That was tested against J01's REAL SUMO phase data
        (not just the hand-typed strings used during initial development)
        and found to be WRONG for this intersection: J01 has two distinct
        always-green links — one on the main junction ('01.01#N0_1') and
        one on the satellite junction ('01.02#S0_1', the bike-crossing
        gate's companion lane from the screenshot). Excluding both
        produced only 8 movements, but phase_pairs references index 8,
        requiring 9. At least one of these always-green links is a real,
        meaningful movement for this model — it just happens to never be
        gated by THIS signal's 4 phases (e.g. one side of a bike/car
        crossing permission that's structurally always-allowed).

        There's no way to tell from phase.state alone which case applies
        (truly-uncontested-and-irrelevant vs always-on-but-meaningful) —
        that's a judgment about traffic semantics, not a green/red
        pattern. So this method does NOT exclude always-green positions
        by default: they get grouped into their own movement(s) like any
        other signature (all all-green positions share the same all-1s
        signature, so they're grouped together as ONE additional
        movement unless you tell this method otherwise). For J01 this
        produces 8 (already-distinct) + 1 (merged all-green) = 9,
        matching phase_pairs.

        *** THIS IS STILL A SIMPLIFICATION, NOT A VERIFIED ANSWER ***
        Merging '01.01#N0_1' (main junction) and '01.02#S0_1' (satellite
        junction) into the same movement index just because they share a
        signature may or may not be semantically correct — they're
        physically different links. The printed output below lists the
        always-green group's lanes explicitly so you can check whether
        that merge is acceptable, or whether phase_pairs intended these
        as two separate movements (in which case you'd need to tell me
        how to split them — there's no automatic way to do so from
        phase.state alone).

        Sets:
            self.movement_index_map: dict[int -> tuple] — movement_index
                -> the green-phase signature defining it, e.g. (1,0,0,1)
                meaning "green in phase 0 and phase 3". A tuple rather
                than a direction string, since there is no general
                cardinal-direction label for an arbitrary signature on an
                irregular junction. The all-green group, if present, uses
                a signature of all 1s (e.g. (1,1,1,1) for a 4-phase
                signal).
            self.movement_lanes: dict[int -> list[str]] — movement_index
                -> controlled-link INBOUND lanes sharing that signature.
                This is what the state fn sums queue/bike_queue over,
                replacing signal.lane_sets[direction] from the earlier
                approach.
            self.movement_out_lanes: dict[int -> list[str]] — movement_index
                -> the corresponding OUTBOUND lanes (the far side of the
                same controlled-link position), derived directly from
                getControlledLinks' (in_lane, out_lane) pairing — NOT from
                lane_sets_outbound's direction-string matching, which uses
                a different, unconnected vocabulary. This lets the state/
                reward functions look up which lanes (and therefore which
                downstream signal) receive traffic from each movement,
                fully generally, to restore the inbound-minus-downstream
                pressure term the original mplight() reward/state used.

        Ordering (still requires your verification): movements are
        numbered by each signature's FIRST occurrence position across
        the phase strings (phase 0 before phase 1, left to right within a
        phase), with the always-green group (if present) placed LAST
        regardless of position — since "always on" doesn't have a
        natural position in the competing-phase ordering, and putting it
        last avoids disturbing the relative order of the genuinely
        phase-varying movements you already verified. There is still no
        way to confirm from code alone that this matches the order
        phase_pairs' 0..K-1 indices were originally intended to mean —
        the printed mapping below (signature AND lanes) is so you can
        check that against the real intersection layout before trusting
        training results.
        """
        flat_inbound = getattr(self, '_flat_inbound_lanes', None)
        if flat_inbound is None:
            raise RuntimeError(
                "_build_movement_index_map requires _build_phase_lane_maps "
                "to have run first (needs self._flat_inbound_lanes)."
            )

        # Outbound lane per position, mirroring flat_inbound's construction
        # but taking the link's SECOND element (out_lane) instead of the
        # first. Built fresh here (rather than reusing flat_inbound) since
        # _build_phase_lane_maps only kept the inbound side.
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

        num_phases = self.num_green_phases
        signature_to_lanes = {}
        signature_to_out_lanes = {}
        signature_first_pos = {}
        all_green_signature = tuple([1] * num_phases) if num_phases > 0 else tuple()

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

            # NOTE: always-green (signature == all 1s) is intentionally
            # NOT excluded here — see docstring above for why an earlier
            # version's exclusion was tested against real J01 data and
            # found to drop a meaningful movement (8 vs the required 9).
            # All always-green positions are grouped together under
            # all_green_signature, same as any other shared signature.

            if signature not in signature_to_lanes:
                signature_to_lanes[signature] = []
                signature_to_out_lanes[signature] = []
                signature_first_pos[signature] = pos
            if lane not in signature_to_lanes[signature]:
                signature_to_lanes[signature].append(lane)

            out_lane = flat_outbound[pos] if pos < len(flat_outbound) else None
            if out_lane and not out_lane.startswith(':') and out_lane not in signature_to_out_lanes[signature]:
                signature_to_out_lanes[signature].append(out_lane)

        # Order: phase-varying signatures by first occurrence, with the
        # always-green group (if present) placed last.
        varying_signatures = sorted(
            (s for s in signature_to_lanes if s != all_green_signature),
            key=lambda s: signature_first_pos[s],
        )
        ordered_signatures = varying_signatures
        if all_green_signature in signature_to_lanes:
            ordered_signatures = varying_signatures + [all_green_signature]

        self.movement_index_map = {i: sig for i, sig in enumerate(ordered_signatures)}
        self.movement_lanes = {i: signature_to_lanes[sig] for i, sig in enumerate(ordered_signatures)}
        self.movement_out_lanes = {i: signature_to_out_lanes[sig] for i, sig in enumerate(ordered_signatures)}

        print(
            f"[{self.id}] derived {len(self.movement_index_map)} movements from "
            f"phase green-signatures: {self.movement_index_map}"
        )
        print(f"[{self.id}] movement -> inbound lanes: {self.movement_lanes}")
        print(f"[{self.id}] movement -> outbound lanes: {self.movement_out_lanes}")
        if all_green_signature in signature_to_lanes:
            always_green_idx = ordered_signatures.index(all_green_signature)
            print(
                f"[{self.id}] *** movement {always_green_idx} is an ALWAYS-GREEN "
                f"group (green in every one of this signal's {num_phases} phases) — "
                f"its lanes are {signature_to_lanes[all_green_signature]}. These "
                f"lanes are merged into ONE movement just because they share a "
                f"signature, which may not be semantically correct if they're "
                f"physically unrelated links (e.g. one on the main junction, one "
                f"on a satellite junction). VERIFY this merge is acceptable before "
                f"trusting training results."
            )
        print(
            f"[{self.id}] >>> VERIFY THIS against the real intersection layout "
            f"before trusting training results — there is no way to confirm "
            f"from code alone that this ordering matches what phase_pairs' "
            f"indices were intended to mean."
        )

    def _build_ped_crossing_groups(self):
        """Group pedestrian-controlled edges into per-direction 'crossings'.

        A crossing roughly corresponds to one crosswalk leg of the intersection
        (north leg, east leg, etc). We key crossings by the same direction
        labels used elsewhere ('N', 'E', 'S', 'W') so they can be tied back to
        phases via phase_ped_lanes.

        self.ped_crossings: dict[direction -> {'in_edges': set, 'out_edges': set}]
            in_edges/out_edges are used as the two 'camera' positions: vehicles
            (here, persons) on in_edges are 'approaching' the crossing, persons
            on out_edges (already across) are 'leaving'.
        """
        self.ped_crossings = {d: {'in_edges': set(), 'out_edges': set()} for d in ('N', 'E', 'S', 'W')}

        links = self.sumo.trafficlight.getControlledLinks(self.id)
        # phase.state position -> link group, same alignment used in
        # _build_phase_lane_maps. We need the *direction* a given pedestrian
        # link belongs to; reuse inbounds_fr_direction's lane membership where
        # possible, else fall back to edge-geometry heuristics already used by
        # generate_config (index_to_movement style) is overkill here — instead
        # we tag a pedestrian link's direction using whichever cardinal lane_set
        # direction shares its connector index, matching how phase_ped_lanes
        # was already built positionally.
        #
        # Practically: for each link group position that has W/w in ANY phase,
        # find the same position's lane via getControlledLinks, then look up
        # which cardinal direction's vehicle lane sits at a 'nearby' index
        # (same index bucket, see infer_phase_pair_from_state's bucket logic)
        # OR — simpler and robust — use the edge's own from/to junction name
        # convention if present, falling back to splitting evenly across W/E/N/S
        # in link order. Below uses the robust positional-bucket approach.

        state_len = max(1, len(links))
        for idx, group in enumerate(links):
            if len(group) == 0:
                continue
            for link in group:
                if len(link) < 2:
                    continue
                in_lane, out_lane = link[0], link[1]
                if in_lane is None or out_lane is None:
                    continue
                # Only pedestrian-relevant lanes: SUMO marks ped lanes via the
                # lane's allowed class, but we don't always have lane.getAllowed
                # cheaply here, so rely on phase_ped_lanes already computed in
                # _build_phase_lane_maps to know which *positions* are 'W'/'w'.
                is_ped_position = False
                for phase_idx in range(self.num_green_phases):
                    state = self.phases[phase_idx].state
                    if idx < len(state) and state[idx] in ('W', 'w'):
                        is_ped_position = True
                        break
                if not is_ped_position:
                    continue

                bucket = min(3, int((idx * 4) / state_len))
                direction = ('N', 'E', 'S', 'W')[bucket]
                try:
                    in_edge = self.sumo.lane.getEdgeID(in_lane)
                    out_edge = self.sumo.lane.getEdgeID(out_lane)
                except Exception:
                    continue
                self.ped_crossings[direction]['in_edges'].add(in_edge)
                self.ped_crossings[direction]['out_edges'].add(out_edge)

        # Drop empty directions so downstream code only iterates real crossings.
        self.ped_crossings = {
            d: v for d, v in self.ped_crossings.items()
            if v['in_edges'] or v['out_edges']
        }

    def _build_phase_pair_ped_crossings(self):
        """Build phase_pair_idx -> [crossing directions walkable in that phase]."""
        self.phase_pair_ped_crossings = {}
        myconfig = signal_configs.get(self.map_name, {})
        valid_acts = myconfig.get('valid_acts', {}).get(self.id, {})

        for pair_idx, local_phase_idx in valid_acts.items():
            ped_lanes = set(self.phase_ped_lanes.get(local_phase_idx, []))
            directions = []
            if ped_lanes:
                for direction, crossing in self.ped_crossings.items():
                    # A crossing is "served" by this phase if any of its in/out
                    # edges' lanes appear in this phase's walk lanes.
                    served = False
                    for lane in ped_lanes:
                        try:
                            edge = self.sumo.lane.getEdgeID(lane)
                        except Exception:
                            continue
                        if edge in crossing['in_edges'] or edge in crossing['out_edges']:
                            served = True
                            break
                    if served:
                        directions.append(direction)
            self.phase_pair_ped_crossings[pair_idx] = directions

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

    def _collect_ped_crossing_pressure(self):
        """Per-crossing 'camera' pressure: approaching_count - leaving_count,
        restricted to persons within self.ped_detect_distance of the crossing
        edges (so it behaves like a fixed-position detector, analogous to
        get_vehicles' max_distance for cars).

        Returns dict[direction -> float], stored on self.ped_crossing_pressure.
        """
        pressure = {}
        for direction, crossing in self.ped_crossings.items():
            approaching = 0
            leaving = 0
            for edge_id in crossing['in_edges']:
                try:
                    person_ids = self.sumo.edge.getLastStepPersonIDs(edge_id)
                except Exception:
                    continue
                for person_id in person_ids:
                    if self._person_within_distance(person_id, edge_id):
                        approaching += 1
            for edge_id in crossing['out_edges']:
                try:
                    person_ids = self.sumo.edge.getLastStepPersonIDs(edge_id)
                except Exception:
                    continue
                for person_id in person_ids:
                    if self._person_within_distance(person_id, edge_id):
                        leaving += 1
            pressure[direction] = max(0.0, float(approaching - leaving))
        self.ped_crossing_pressure = pressure
        return pressure

    def _person_within_distance(self, person_id, edge_id):
        """Camera-style cutoff: only count a person if they're within
        self.ped_detect_distance of the edge (using lane position along the
        edge as a simple proxy, since persons don't have getNextTLS like
        vehicles do)."""
        try:
            lane_pos = self.sumo.person.getLanePosition(person_id)
            edge_len = self.sumo.lane.getLength(edge_id + '_0')
        except Exception:
            return True  # fail open: if we can't measure, don't silently drop them
        # Distance to the *end* of the edge (i.e. distance to the crossing).
        distance_to_crossing = max(0.0, edge_len - lane_pos)
        return distance_to_crossing <= self.ped_detect_distance

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