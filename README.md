# T-REX

<p align="left">
  <img src="TREX-light.png" alt="T-REX" width="480"/>
</p>

![CI](https://github.com/andngdtudk/T-REX/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2506.13836-b31b1b.svg)](https://arxiv.org/abs/2506.13836)

**T-REX** (Traffic control Environment for Robustness Evaluation under incidents) is a
[SUMO](https://www.eclipse.org/sumo/)-based simulation framework for training and
evaluating Reinforcement-Learning-based Traffic Signal Control (RL-TSC) methods under
network-level incident conditions — collisions, lane blockages, stalled vehicles — rather
than only the free-flow conditions most RL-TSC benchmarks assume. It injects incidents
with configurable location/duration/severity, simulates realistic driver responses
(rerouting, speed adaptation, lane changing) around them, and exposes a Gym-compatible
interface so any RL-TSC method can be trained and evaluated on how much its performance
degrades when the network is disrupted.

> Companion code for *"Robustness of Reinforcement Learning-Based Traffic Signal Control
> under Incidents: A Comparative Study"* (see [Citation](#citation)).

**Before you rely on any number this repository produces**, read
[`AUDIT_REPORT.md`](AUDIT_REPORT.md) — an independent code audit that found, among other
things, that the incident scenario currently cannot run on any of the four real-world
networks from a clean checkout (the incident-location-probability CSVs it depends on are
missing from the repo), and that this repo does not itself implement the paper's
LSI/FPD/CR/AUC/RAUC/PDI robustness metrics. Both are flagged there in detail.

## Key features

- **Incident-aware simulation** — inject a lane-blocking incident at a chosen or randomly
  sampled edge/position/lane-count/duration/start-time (Section 2.3).
- **Information Comply Model (ICM) rerouting** — a probabilistic, awareness-based model of
  whether and when a driver reroutes around a known incident (radio/VMS/app/observation
  awareness sources feeding a binomial-logit rerouting decision; Appendix A).
- **Stopping-Sight-Distance (SSD) speed adaptation** — vehicles approaching a blocked lane
  slow down once the incident is within their AASHTO stopping sight distance (Section 2.4.2).
- **Contextual lane-changing** — modified SUMO LC2013 lane-change parameters near an
  incident (Section 2.4.3).
- **Gym-compatible RL interface**, adapted from [RESCO](https://github.com/Pi-Star-Lab/RESCO),
  supporting **IDQN, IPPO, MPLight, FMA2C**, and rule-based baselines (Fixed-time, Random,
  Max-pressure, Greedy).
- **Modular by design** — the incident model (`Initializer`/`Deployment`) is decoupled from
  the RL interface, so it's meant to be portable to other traffic simulators or RL-TSC
  stacks, not only this repo's SUMO+RESCO integration.

## Architecture

T-REX is organized into four conceptual modules (the paper's Figure 1); in this repository
they map onto code as follows:

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Network Environment │────▶│    Initializer    │────▶│      Deployment       │────▶│    RL Interaction     │
│                      │     │                    │     │                       │     │                       │
│ base_env.py          │     │ T_REX.py::         │     │ T_REX.py::            │     │ TREX_comp/            │
│ incident_env.py      │     │  Initializer       │     │  Deployment           │     │  agents/, rewards.py, │
│ traffic_signal.py    │     │                    │     │                       │     │  states.py, config/   │
│ environments/*       │     │ samples edge/lanes/ │     │ injects the incident   │     │ + main.py entry point │
│ (SUMO network,       │     │ position/duration/  │     │ via TraCI, runs ICM     │     │                       │
│  routes, signal      │     │ start time          │     │ rerouting + SSD speed   │     │ Gym-style state/      │
│  plans, vtypes)      │     │                    │     │ adaptation + lane-      │     │ action/reward loop     │
│                      │     │                    │     │ changing each step      │     │                       │
└─────────────────────┘     └──────────────────┘     └──────────────────────┘     └─────────────────────┘
```

`IncidentEnv` (in `incident_env.py`) wires these together: each episode it constructs an
`Initializer` (sampling or accepting fixed incident parameters), and if an incident is
active, a `Deployment` that runs the rerouting/speed/lane-change logic once per simulation
step. `BaseEnv` (`base_env.py`) is the incident-free counterpart used for `--strategy 1`
training. Both expose the same `reset()`/`step()`/`close()` Gym interface that
`TREX_comp`'s agents and `main.py`'s training loop consume.

## Installation

### 1. SUMO

Install [SUMO](https://www.eclipse.org/sumo/) and set `SUMO_HOME` to its installation
directory (required by `sumolib`/`traci`). This repository was developed and smoke-tested
against **SUMO 1.27**; older 1.19+ releases are likely compatible but haven't been verified.

```bash
# Debian/Ubuntu
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo   # add to your shell profile
```

Alternatively, `pip install eclipse-sumo` (already in `requirements.txt`) bundles the SUMO
binaries as a Python package, which is what the CI/smoke-test environment for this audit
used.

### 2. Python environment

Python 3.10+ (tested on 3.13). Using a virtual environment is recommended:

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda create -n trex python=3.11
pip install -r requirements.txt
```

To run libsumo (faster, single-process SUMO) instead of TraCI, set
`LIBSUMO_AS_TRACI=1` in your environment and pass `--libsumo True` to `main.py`
(this is `main.py`'s default).

### 3. RESCO

`TREX_comp/` is a self-contained adaptation of [RESCO](https://github.com/Pi-Star-Lab/RESCO)'s
agent/environment interface (see `base_env.py:2`) — you do **not** need to install RESCO
separately to train or evaluate with `main.py`. The `resco_benchmark` package is imported
only by the standalone analysis script `readXML.py`, which is not part of the training
pipeline; install it from source (`pip install git+https://github.com/Pi-Star-Lab/RESCO`)
only if you need that specific script.

### 4. Verify the install

```bash
# Decompress the grid4x4 traffic-flow files (only needed for grid4x4/arterial4x4)
cd environments/grid4x4 && unzip grid4x4.zip -d grid4x4 && cd ../..

export LIBSUMO_AS_TRACI=1
python main.py --agent MAXPRESSURE --map grid4x4 --eps 1 --strategy 1 --libsumo True
```

This runs one episode of a rule-based Max-pressure controller on Grid4x4 with no incidents
and writes `results/.../metrics_1.csv`. If it completes without a traceback, your SUMO/Python
setup is working. Run the unit tests as a second check (these don't need SUMO installed):

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Quick start

```bash
export LIBSUMO_AS_TRACI=1

# Max-pressure baseline on Grid4x4, no incidents
python main.py --agent MAXPRESSURE --map grid4x4 --eps 10 --strategy 1 --libsumo True

# Same, but with incidents injected (Initializer/Deployment active)
python main.py --agent MAXPRESSURE --map grid4x4 --eps 10 --strategy 2 --libsumo True
```

`--strategy 1` uses `BaseEnv` (no incidents); `--strategy 2` uses `IncidentEnv` with
randomly sampled incidents each episode. **`--strategy 3` ("curriculum") is present in the
CLI help text but not actually implemented — it currently crashes; see `AUDIT_REPORT.md`
Section 2.6b.**

Swap `--agent` for any of `STOCHASTIC`, `MAXWAVE`, `MAXPRESSURE`, `IDQN`, `IPPO`, `MPLight`,
`FMA2C`, and `--map` for `grid4x4`, `arterial4x4`, `ingolstadt1/7/21`, `cologne1/3/8`.
IDQN/IPPO/MPLight/FMA2C additionally require `torch`, `tensorflow`, and `pfrl` (all in
`requirements.txt`).

## Reproducing the paper's experiments

The paper reports three experiments (learning performance, testing/generalization, and
transferability/online adaptation) plus a Table 1 runtime/scalability benchmark. This
repository has a **single entry point** (`main.py`) rather than one script per experiment;
each experiment corresponds to a particular combination of CLI flags:

| Experiment | Flags | Notes |
|---|---|---|
| 1 — Learning performance | `--strategy 2 --agent <method> --map <network> --eps <N>` | Trains one RL-TSC method under incidents from scratch; repeat per method/network. |
| 2 — Testing/generalization | `--strategy 2 --load True --agent <method> --map <network>` | Loads a trained model (`--load True`) and runs held-out episodes without further training. |
| 3 — Transferability/online adaptation | `--strategy 2 --repeat <N> --load True` | `--repeat` saves (during training) and later replays (during testing) a fixed set of the last `N` incident seeds, for evaluating a trained agent against a controlled, reproducible incident set — see `main.py::run_incident_scenario`. |
| Table 1 — runtime/scalability | any of the above | Wall-clock/episode is not separately instrumented in this repo; time your own runs per network. |

**Known reproducibility gaps** (see `AUDIT_REPORT.md` for the full analysis — do not treat
this table as a guarantee that these commands reproduce the paper's published numbers):
the per-network learning-rate defaults from Appendix B are only wired up for IDQN/MPLight
(`TREX_comp/config/hyperparams.py`); SUMO is always launched with `--random` rather than a
controlled `--seed`, so full seed-for-seed reproduction of a specific run isn't currently
possible even where the incident seed is fixed; and the incident-location-probability CSVs
required for weighted sampling on Ingolstadt/Cologne are missing from the repo entirely
(§2.8 of the audit report), so `--strategy 2` cannot run on those four networks from a
clean checkout until that data is restored.

## Configuring incidents

`Initializer` supports two modes:

**Random sampling** (used by `IncidentEnv` each episode): edge (weighted by historical flow
data for Ingolstadt/Cologne networks, uniform otherwise), number of blocked lanes, position
along the edge (`U(10, edge_length−10)`), start time, and duration
(`~Exponential(rate=0.029)` minutes) are all drawn automatically — see `Initializer.random()`.

**Fixed/user-defined incident**, via `Initializer.set_incident(...)`:

```python
from T_REX import Initializer

incident = Initializer(map_name="grid4x4", run_num=0, scenario_folder="environments/grid4x4/grid4x4.net.xml",
                        warm_up_time=0, end_time=3600)
incident.set_incident(
    edge="A2A1",         # SUMO edge ID
    lanes=[0, 1],        # which lanes are blocked
    pos=150.0,           # position along the edge, in meters
    start_time=600,      # simulation second the incident begins
    duration=300,        # how many seconds it lasts
    is_incident=True,
)
```

Note: the current implementation models incidents as a single generic "N lanes of an edge
become impassable" mechanism — it does **not** have a selectable `incident_type` parameter
distinguishing the five categories the paper's Table B1 describes (collision, stalled
vehicle, roadworks, speed-reduction/environmental, signal malfunction). Any of those can be
*represented* by the generic mechanism above (e.g. a full-width blockage for a collision, a
single-lane blockage for a stalled vehicle), but there's no code-level switch to pick
between them — see `AUDIT_REPORT.md` Section 2.4b.

## Repository structure

```
T_REX.py                 Initializer + Deployment: incident sampling, ICM rerouting,
                          SSD speed adaptation, lane-changing, teleport-exemption
base_env.py               Gym env, no incidents (--strategy 1)
incident_env.py            Gym env with incidents (--strategy 2), wraps Initializer/Deployment
traffic_signal.py          Per-intersection Signal class (phases, observations)
main.py                    CLI entry point / training loop
graph.py, readCSV.py,       Ad hoc analysis/plotting scripts (not part of the core pipeline;
readXML.py                  readXML.py needs a separately-installed RESCO, see Installation)
TREX_comp/
  agents/                   RL-TSC method implementations (IDQN, IPPO, MPLight, FMA2C,
                              Max-pressure, Greedy/MAXWAVE, Random/STOCHASTIC)
  config/                    agent_config.py, map_config.py, mdp_config.py, signal_config.py,
                              hyperparams.py (per-network learning-rate defaults)
  rewards.py, states.py      Reward functions and observation functions per method
environments/               SUMO network/route/additional files per benchmark network
tests/                      pytest unit tests (SSD, ICM, incident sampling) -- no SUMO needed
.github/workflows/ci.yml    CI: pytest (required) + ruff/black/isort (advisory)
AUDIT_REPORT.md             Independent code audit: findings, fixes, and open issues
CHANGELOG.md                What changed in the audit, by phase
```

## Supported networks

| Network | Source |
|---|---|
| `grid4x4` | Synthetic 4×4 grid, 16 intersections |
| `arterial4x4` | Synthetic 4×4 arterial network (from RESCO's benchmark suite; not one of the paper's four evaluation networks) |
| `cologne3` (Corridor) / `cologne8` (Region) | [TAPAS Cologne](https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html) |
| `ingolstadt7` (Corridor) / `ingolstadt21` (Region) | [InTAS](https://github.com/silaslobo/InTAS) |
| `cologne1`, `ingolstadt1` | Single-intersection reductions of the above, useful for quick local testing |

To add a new network: add a `.sumocfg`/`.net.xml` (+ `.add.xml` if you need incident vTypes
like `CAV1`/`CAV3`/`IC`) under `environments/<name>/`, and an entry in
`TREX_comp/config/map_config.py` (step length, yellow length, start/end time, warm-up).
`signal_configs` in `TREX_comp/config/signal_config.py` needs a `phase_pairs`/`valid_acts`
entry if you intend to use MPLight/FMA2C on it.

## Extending T-REX

- **New RL-TSC method**: add an agent class under `TREX_comp/agents/` (see `agent.py` for the
  `Agent`/`IndependentAgent`/`SharedAgent` base classes), a reward function in `rewards.py`
  and/or state function in `states.py` if it needs a custom MDP, and an entry in
  `TREX_comp/config/agent_config.py`.
- **A different simulator**: the incident model (`Initializer`/`Deployment` in `T_REX.py`)
  is designed to be decoupled from the RL interface — porting it means reimplementing the
  small set of TraCI calls it makes (vehicle position/speed/route/type queries and
  edits) against your simulator's equivalent API; the sampling distributions, ICM/SSD math,
  and lane-change parameter logic don't otherwise depend on SUMO.

## Citation

```bibtex
@misc{nguyen2025robustnessreinforcementlearningbasedtraffic,
      title={Robustness of Reinforcement Learning-Based Traffic Signal Control under Incidents: A Comparative Study},
      author={Dang Viet Anh Nguyen and Carlos Lima Azevedo and Tomer Toledo and Filipe Rodrigues},
      year={2025},
      eprint={2506.13836},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.13836},
}
```

This paper has been submitted to *European Transport Research Review*; journal
volume/pages/DOI are not yet assigned (TBD — see `CITATION.cff`). The arXiv preprint above
is the citable version in the meantime.

## License

The T-REX source code is released under the [MIT License](LICENSE). The traffic network
datasets under `environments/` are third-party research data redistributed under their own
licenses — see `environments/LICENSE` and the per-network Creative Commons notices.

## Acknowledgments

This project incorporates components adapted from the open-source
[RESCO](https://github.com/Pi-Star-Lab/RESCO) repository (Pi-Star Lab), which provides a
Gym-compatible benchmarking environment for RL-based traffic signal control on SUMO. Thanks
to the authors for their contribution to the research community.

## Contact / maintainers

For bugs or questions, please open a [GitHub issue](https://github.com/andngdtudk/T-REX/issues).
See `CONTRIBUTING.md` for how to submit a pull request.
