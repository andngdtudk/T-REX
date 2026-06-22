## IDQN Multimodal Extension: Development Log

### Starting Point

The baseline `IDQN_DELTASCLIP` agent successfully reduced car waiting times on `kbh_j1_442m`. The goal was to extend it to optimize across cars, bikes, and pedestrians simultaneously, using a new `IDQN_MM2` configuration with:
- State: `drq_mm2` — 9 features per lane including bike and ped wait
- Reward: `wait_multimodal_delta_sclip` — delta of weighted combined wait
- Initial config: `norm_wait=224`, `clip_wait=4`, equal weights, `EPS_DECAY=220`, `TARGET_UPDATE=500`

---

### Issue 1 — Reward Saturation (`norm_wait` too small)

**Symptom:** Mean reward locked at exactly ±4.0 every episode. All waits increasing. Q-values collapsed from 31 to 7 (−77%).

**Diagnosis:** `norm_wait=224` was inherited from the single-mode agent where `total_wait` is per-lane. With bikes accumulating ~150M s·steps per episode, per-step deltas were orders of magnitude larger than 224, so every reward was clipped to ±4. The agent received a binary signal with no gradient information.

**Fix:** Built `calibrate_norm_wait.py` to compute the correct `norm_wait` from actual step deltas in `mm_step_metrics.csv`. The calibrator recommended `norm_wait=441`, `clip_wait=3`. This unclipped the reward (range became −2.97 to +3.0) and produced the first run with genuine learning: car −25%, bike −25%, queues −15–19%, but +300% for pedestrians. Weights for all modes are 1.0.

---

### Issue 2 — Q-value Divergence (gradient explosion)

**Symptom:** Q-values exploding to 124 million, loss to 2 million, after changing mode weights. Network in freefall.

**Diagnosis:** pfrl's `DQN` constructor accepts `max_grad_norm` but it was not being passed, defaulting to `None` (no clipping). Each reward scale change also shifted what Q=100 meant, destabilizing value estimates. Additionally, stale replay buffer entries from previous reward scales mixed with new ones.

**Fix:** Added `max_grad_norm=config.get('MAX_GRAD_NORM', 10.0)` to both `DQN()` and `SharedDQN()` constructor calls in `DQNAgent.__init__`. Added `MAX_GRAD_NORM` to agent config (later reduced to 1.0 when Q still grew to ~13,000). Added `clear_replay_buffer()` method to `DQNAgent` and `IDQN` for use when restarting with a new reward scale.

---

### Issue 3 — Premature Policy Convergence (`EPS_DECAY` too small)

**Symptom:** Q-values spiked then collapsed around episode 40 in every run. Policy stagnated early. Loss eventually near zero but waits not improving.

**Diagnosis:** `EPS_DECAY=220` caused epsilon to reach `EPS_END` after approximately 0.3 episodes (220 / 720 decisions per episode). The agent was essentially fully greedy from the start, reinforcing whatever random initial policy it stumbled into before the replay buffer contained useful experience. The Q-spike at episode ~35–40 coincided with the replay buffer filling and cycling, causing abrupt target shifts.

**Fix:** Calculated correct decay schedule: 720 decisions/episode × 100 episodes × 0.7 = 50,400. Set `EPS_DECAY=50400`, `EPS_END=0.05`, `GAMMA=0.95`. Switched from hard target updates (`TARGET_UPDATE=500`) to soft updates (`target_update_interval=1`, `target_update_method='soft'`, `soft_update_tau=0.005`, later increased to 0.02) to prevent the bootstrap target from becoming stale between hard sync events.

---

### Issue 4 — Wrong Convolutional Architecture

**Symptom:** Stable numerics but no learning. Loss near zero, Q negative and flat, waits not improving across multiple runs with different hyperparameters.

**Diagnosis:** Inspecting `obs_act` revealed observation shape `(1, 21, 9)` — 1 channel, 21 lanes, 9 features. The existing `Conv2d(1, 64, kernel_size=(2,2))` treated the observation as a 2D spatial grid, convolving across both lane and feature dimensions simultaneously. This mixed heterogeneous features (e.g. `car_wait` from lane N with `bike_wait` from lane N+1), extracting meaningless cross-feature patterns. The car-only agent's smaller, more homogeneous feature set was less sensitive to this flaw.

**Fix:** Replaced the convolutional model with `LaneWiseModel` — a lane-wise encoder that applies a shared `Linear(9→128→128)` network independently to each lane's feature vector, then mean-pools across all 21 lanes before the action head. This correctly treats lanes as exchangeable units with a coherent 9-feature description, rather than a 2D spatial grid.

### Issue 5 — Episode-Boundary Reward Poisoning

---

**Symptom:** Debug logging revealed `reward=+3.0` on decision 1 of every episode with `car_wait=0, bike_wait=0`. Persistent negative learning trend despite other fixes.

**Diagnosis:** `wait_multimodal_delta_sclip` stores `_prev_waits` across episodes. At the episode boundary, the environment resets to zero wait, but a bug in the reward function caused it to hold the large accumulated wait from the final step of the previous episode. The first delta therefore becomes `0 - large_number`, which after negation yields a large spurious positive reward. This poisoned the replay buffer with incorrect Q-targets on every episode's first transition.

**Fix:** Added explicit reset calls in `run_episode` immediately after `env.reset()`:
```python
if hasattr(env.state_fn, 'reset'):
    env.state_fn.reset()
if hasattr(env.reward_fn, 'reset'):
    env.reward_fn.reset()
```

---

### Issue 6 — Uncontrollable Bike Wait in Reward

**Symptom:** Even after fixes 1–6, bike wait comprised 73% of the combined reward signal but per-phase debug logging showed identical bike_wait across all 4 phases.

**Diagnosis:** Bikes share lanes with cars at this intersection — all phases serve bikes equally. The agent had no lever to reduce bike wait through phase selection. Including `bike_wait` in the reward added ~72% noise to the learning signal with zero useful gradient: the agent was being penalized for something structurally outside its control.

**Fix:** Set `bike_wait_weight=0.0` in the reward config while retaining bike features in the state (`drq_mm2_delta` features 3 and 5). The agent can observe bike conditions but is no longer penalized for them. Bike wait is tracked as a monitoring metric only.

---

### First Successful Run

After removing the bike wieght:
- Car wait: **−15.1%** (first-10 vs last-10 episodes)
- Bike wait: **−21.0%** (emergent — agent learned car phases that incidentally help bikes)
- Car queue: **−4.8%**
- Bike queue: **−10.1%**
- Mean reward: **+29.7%** (trending toward zero)

Remaining open items: `norm_wait` needs recalibration for the new `bike_weight=0` scale (reward still clipping at ±3); ped wait slightly increasing (+31.8%) and may benefit from `ped_wait_weight` adjustment after recalibration; Q-values still drifting negative suggesting further `soft_update_tau` tuning.