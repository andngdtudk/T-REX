"""
Standalone test of FRAP_MM.forward's tensor mechanics, with the
DiscreteActionValueHead and SharedAgent/pfrl imports stripped out
(those aren't installed in this sandbox and aren't relevant to the
shape logic we're checking).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FRAP_MM_Test(nn.Module):
    def __init__(self, demand_shape, output_shape, phase_pairs, competition_mask, device, num_movements):
        super().__init__()
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        self.num_movements = num_movements
        self.comp_mask = competition_mask
        self.device = device
        self.demand_shape = demand_shape

        self.d_out = 4
        self.p_out = 4
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)
        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)
        self.ped_embed = nn.Linear(1, self.lane_embed_units)
        self.lane_conv = nn.Conv2d(2 * self.lane_embed_units, 20, kernel_size=(1, 1))
        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))
        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

    def forward(self, states):
        states = states.to(self.device)
        batch_size = states.size()[0]
        acts = states[:, 0].to(torch.int64)
        rest = states[:, 1:].float()

        movement_block_len = self.num_movements * self.demand_shape
        expected_len = movement_block_len + self.oshape
        actual_len = rest.size(1)
        assert actual_len == expected_len, (actual_len, expected_len)

        movement_states = rest[:, :movement_block_len]
        ped_block = rest[:, movement_block_len:]

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

        ped_embeds = F.relu(self.ped_embed(ped_block.unsqueeze(-1)))

        pairs = []
        for pair_idx, pair in enumerate(self.phase_pairs):
            combined = phase_demands[:, pair[0]] + phase_demands[:, pair[1]]
            combined = combined + ped_embeds[:, pair_idx]
            pairs.append(combined)

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j:
                    rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
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
        return q_values  # skip DiscreteActionValueHead, just check raw shape


if __name__ == "__main__":
    # Mimic a 4-way intersection: 8 movements (N/E/S/W in/out... simplified
    # to match typical grid4x4-style phase_pairs), 4 phase pairs.
    phase_pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
    num_movements = 8
    demand_shape = 2  # car, bike
    num_actions = len(phase_pairs)

    comp_mask = []
    for i in range(len(phase_pairs)):
        zeros = torch.zeros(len(phase_pairs) - 1, dtype=torch.int64)
        cnt = 0
        for j in range(len(phase_pairs)):
            if i == j:
                continue
            pair_a, pair_b = phase_pairs[i], phase_pairs[j]
            if len(set(pair_a + pair_b)) == 3:
                zeros[cnt] = 1
            cnt += 1
        comp_mask.append(zeros)
    comp_mask = torch.stack(comp_mask)

    model = FRAP_MM_Test(demand_shape, num_actions, phase_pairs, comp_mask, torch.device('cpu'), num_movements)

    batch_size = 5
    state_len = 1 + num_movements * demand_shape + num_actions
    states = torch.zeros(batch_size, state_len)
    states[:, 0] = torch.randint(0, num_actions, (batch_size,))  # action/phase index
    states[:, 1:] = torch.rand(batch_size, state_len - 1)

    out = model(states)
    print("Output shape:", out.shape, "expected:", (batch_size, num_actions))
    assert out.shape == (batch_size, num_actions)

    # Sanity: changing ONLY the pedestrian block should change the
    # Q-values (i.e. ped_embed is actually wired into the graph, not a
    # dead branch).
    states2 = states.clone()
    ped_start = 1 + num_movements * demand_shape
    states2[:, ped_start:] = states2[:, ped_start:] + 5.0  # bump ped pressure a lot
    out2 = model(states2)
    diff = (out - out2).abs().sum().item()
    print("Sum abs diff after perturbing ONLY ped block:", diff)
    assert diff > 1e-6, "Pedestrian block had no effect on output — embedding not wired correctly!"

    # Sanity: changing ONLY the bike portion of one movement should also matter.
    states3 = states.clone()
    states3[:, 2] = states3[:, 2] + 5.0  # movement 0's bike feature (car=idx1, bike=idx2)
    out3 = model(states3)
    diff_bike = (out - out3).abs().sum().item()
    print("Sum abs diff after perturbing ONLY one movement's bike feature:", diff_bike)
    assert diff_bike > 1e-6, "Bike feature had no effect on output!"

    print("ALL SHAPE/WIRING CHECKS PASSED")

    # --- Second case: a movement index never referenced by any phase pair ---
    # (e.g. a permanently-protected right turn). The OLD inference approach
    # (max referenced index + 1) would have computed num_movements=6 here,
    # silently wrong since there are actually 7 movements (0..6) and movement
    # 6 is just never paired. Explicit num_movements=7 must be used instead.
    phase_pairs_gap = [[0, 4], [1, 5], [2, 3]]  # never references movement 6
    num_movements_gap = 7  # ground truth, e.g. from len(signal.lane_sets)
    num_actions_gap = len(phase_pairs_gap)

    comp_mask_gap = []
    for i in range(len(phase_pairs_gap)):
        zeros = torch.zeros(len(phase_pairs_gap) - 1, dtype=torch.int64)
        cnt = 0
        for j in range(len(phase_pairs_gap)):
            if i == j:
                continue
            pair_a, pair_b = phase_pairs_gap[i], phase_pairs_gap[j]
            if len(set(pair_a + pair_b)) == 3:
                zeros[cnt] = 1
            cnt += 1
        comp_mask_gap.append(zeros)
    comp_mask_gap = torch.stack(comp_mask_gap)

    model_gap = FRAP_MM_Test(demand_shape, num_actions_gap, phase_pairs_gap, comp_mask_gap,
                              torch.device('cpu'), num_movements_gap)

    state_len_gap = 1 + num_movements_gap * demand_shape + num_actions_gap
    states_gap = torch.zeros(2, state_len_gap)
    states_gap[:, 0] = torch.randint(0, num_actions_gap, (2,))
    states_gap[:, 1:] = torch.rand(2, state_len_gap - 1)

    out_gap = model_gap(states_gap)
    print("Gap-case output shape:", out_gap.shape, "expected:", (2, num_actions_gap))
    assert out_gap.shape == (2, num_actions_gap)
    print("GAP-CASE (unreferenced movement) CHECK PASSED — explicit num_movements required for this to work")
