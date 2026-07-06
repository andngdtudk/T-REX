import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from TREX_comp.agents.agent import SharedAgent
from TREX_comp.agents.pfrl_dqn import DQNAgent
from TREX_comp.config.signal_config import signal_configs
from pfrl.q_functions import DiscreteActionValueHead


class MPLight_MM(SharedAgent):
    """MPLight extended with bicycle and pedestrian awareness.

    Bicycles are folded into the existing per-movement demand vector
    (demand_shape=2: car_pressure, bike_pressure) — structurally identical
    to vanilla MPLight/FRAP, just a wider per-movement feature.

    Pedestrians are NOT a movement (a phase pair doesn't "own" a crossing
    the way it owns two vehicle movements), so pedestrian pressure is
    injected at the phase-PAIR level, once per phase_pairs entry, via a
    learned embedding added into FRAP's pairwise competition graph right
    where pairs are already phase-indexed (see FRAP_MM.forward).

    State vector layout expected from state fn `mplight_mm`:
        [phase,
         m0_car, m0_bike, m1_car, m1_bike, ..., m{K-1}_car, m{K-1}_bike,
         p0_ped, p1_ped, ..., p{N-1}_ped]
    where K = num_movements (len(signal.lane_sets), must be passed
    explicitly via config['num_movements'] or
    signal_configs[map_name]['num_movements'] — see __init__) and
    N = num_actions = len(phase_pairs) (global, see state_mplight_mm.py
    for why N must be global/padded rather than per-signal).
    """

    def __init__(self, config, obs_act, map_name, thread_number, lr=0.005):
        super().__init__(config, obs_act, map_name, thread_number)
        phase_pairs = signal_configs[map_name]['phase_pairs']
        num_actions = len(phase_pairs)

        # num_movements: must equal len(signal.lane_sets) for every signal
        # on this map (same implicit assumption vanilla MPLight already
        # made — phase_pairs indices have to mean the same thing for every
        # shared signal). We require it explicitly in config rather than
        # inferring it, since silent mis-inference here corrupts the
        # movement/pedestrian split in FRAP_MM.forward with no error.
        #
        # If you don't already have this in signal_configs[map_name], add it
        # (e.g. 'num_movements': 8 alongside 'phase_pairs'/'valid_acts'), or
        # pass it via config['num_movements'] when constructing the agent.
        num_movements = config.get('num_movements') or signal_configs[map_name].get('num_movements')
        if num_movements is None:
            raise ValueError(
                f"MPLight_MM requires 'num_movements' for map '{map_name}' "
                f"(either in config or signal_configs[map_name]) — this must "
                f"equal len(signal.lane_sets) for every signal on this map, "
                f"matching how state_mplight_mm.mplight_mm() lays out the "
                f"per-movement demand block."
            )

        # num_movements may now be dict[signal_id -> int] (movement count
        # can legitimately differ per signal). SharedAgent batches every
        # signal through one shared model, so we need a single global
        # width: take the max here, and rely on the state fn
        # (mplight_mm in state_mplight_mm.py) zero-padding any shorter
        # signal's state vector up to this same width — same approach
        # already used for the pedestrian block, for the same reason.
        if isinstance(num_movements, dict):
            num_movements = max(num_movements.values())

        comp_mask = []
        for i in range(len(phase_pairs)):
            zeros = np.zeros(len(phase_pairs) - 1, dtype=int)
            cnt = 0
            for j in range(len(phase_pairs)):
                if i == j: continue
                pair_a = phase_pairs[i]
                pair_b = phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = np.asarray(comp_mask)
        print(comp_mask)

        comp_mask = torch.from_numpy(comp_mask).to(self.device)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        model = FRAP_MM(config, num_actions, phase_pairs, comp_mask, self.device, num_movements)
        self.agent = DQNAgent(config, num_actions, model, num_agents=config['num_lights'], lr=lr)
        if self.config['load']:
            print('LOADING SAVED MODEL FOR EVALUATION')
            self.agent.load(self.config['log_dir'] + 'agent.pt')
            self.agent.agent.training = False


class FRAP_MM(nn.Module):
    """FRAP extended with car+bike per-movement demand and a phase-pair
    level pedestrian embedding injected into the competition graph.

    Differences from the original FRAP:
      - demand_shape is taken from config (expected: 2, for [car, bike])
        rather than assumed to be a single queue value. The demand
        Linear/embedding layers are otherwise unchanged in spirit — they
        just take a wider input now.
      - The state is split into a per-movement block (size
        num_movements * demand_shape) and a per-phase-pair pedestrian
        block (size oshape == len(phase_pairs)), using an EXPLICIT split
        point rather than inferring num_movements arithmetically. This
        is required because the pedestrian block has a fixed length
        (oshape) independent of demand_shape/num_movements, so the
        original "(total_len - 1) / demand_shape" formula no longer
        identifies num_movements correctly once a ped block is appended.
      - A small ped_embed layer maps each phase pair's scalar pedestrian
        pressure to a lane_embed_units-sized vector, which is ADDED to
        that pair's combined demand embedding (the `pairs[i]` term) right
        before the rotated-phases competition step. This is the layer
        where FRAP first becomes phase-pair-indexed (each `pairs[i]`
        already corresponds to phase_pairs[i]), so it's the natural place
        for phase-pair-level pedestrian context to enter — earlier layers
        are indexed by movement, not phase pair, and pedestrians aren't a
        movement.
    """

    def __init__(self, config, output_shape, phase_pairs, competition_mask, device, num_movements):
        super(FRAP_MM, self).__init__()
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        # num_movements MUST be passed explicitly (= len(signal.lane_sets)
        # for this map, the same quantity state_mplight_mm.mplight_mm()
        # iterates over) rather than inferred from phase_pairs. Inferring
        # it as max(index referenced in phase_pairs)+1 silently breaks if
        # some movement index is never the target of any phase pair —
        # which is a real possibility (e.g. a permanently-protected
        # right-turn movement with no competing phase), and the failure
        # mode is a wrong split point between the movement block and the
        # pedestrian block with NO error raised, just garbage features.
        self.num_movements = num_movements
        self.comp_mask = competition_mask
        self.device = device

        # demand_shape now describes the per-movement vehicle-demand width
        # (car + bike = 2), NOT including the pedestrian block.
        self.demand_shape = config.get('demand_shape', 2)

        self.d_out = 4
        self.p_out = 4
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)
        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        # Pedestrian pressure -> embedding added to each phase pair's
        # combined demand (2*lane_embed_units wide, matching `pairs[i]`
        # after the i,j concat happens later — see note in forward()).
        self.ped_embed = nn.Linear(1, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2 * self.lane_embed_units, 20, kernel_size=(1, 1))

        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))

        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        states = states.to(self.device)
        batch_size = states.size()[0]
        acts = states[:, 0].to(torch.int64)
        rest = states[:, 1:].float()

        # Explicit split: [movement_block (num_movements * demand_shape)] [ped_block (oshape)]
        movement_block_len = self.num_movements * self.demand_shape
        expected_len = movement_block_len + self.oshape
        actual_len = rest.size(1)
        if actual_len != expected_len:
            raise ValueError(
                f"FRAP_MM got state width {actual_len}, expected "
                f"{expected_len} (= num_movements[{self.num_movements}] * "
                f"demand_shape[{self.demand_shape}] + oshape[{self.oshape}]). "
                f"Check that the state fn (mplight_mm) and this model agree "
                f"on num_phase_pairs / demand_shape."
            )

        movement_states = rest[:, :movement_block_len]
        ped_block = rest[:, movement_block_len:]  # (batch, oshape) one scalar per phase pair

        # Expand action index to mark demand input indices (unchanged from FRAP)
        extended_acts = []
        for i in range(batch_size):
            act_idx = int(acts[i].item())
            if act_idx < 0 or act_idx >= len(self.phase_pairs):
                act_idx = act_idx % len(self.phase_pairs)
            pair = self.phase_pairs[act_idx]
            zeros = torch.zeros(self.num_movements, dtype=torch.int64, device=self.device)
            zeros[pair[0]] = 1
            zeros[pair[1]] = 1
            extended_acts.append(zeros)
        extended_acts = torch.stack(extended_acts)
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        for i in range(self.num_movements):
            phase = phase_embeds[:, i]
            demand = movement_states[:, i * self.demand_shape:(i + 1) * self.demand_shape]
            demand = torch.sigmoid(self.d(demand))
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

        # Pedestrian embedding per phase pair, added into that pair's
        # combined movement-demand embedding. This is what lets a phase
        # pair's score go up when it would serve waiting pedestrians,
        # without pedestrians needing to be a "movement".
        ped_embeds = F.relu(self.ped_embed(ped_block.unsqueeze(-1)))  # (batch, oshape, lane_embed_units)

        pairs = []
        for pair_idx, pair in enumerate(self.phase_pairs):
            combined = phase_demands[:, pair[0]] + phase_demands[:, pair[1]]
            combined = combined + ped_embeds[:, pair_idx]
            pairs.append(combined)

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(
            rotated_phases,
            (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units),
        )
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)
        rotated_phases = F.relu(self.lane_conv(rotated_phases))

        competition_mask = self.comp_mask.tile((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)
        relations = F.relu(self.relation_conv(relations))

        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))
        combine_features = self.before_merge(combine_features)

        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = torch.sum(combine_features, dim=-1)
        return self.head(q_values)