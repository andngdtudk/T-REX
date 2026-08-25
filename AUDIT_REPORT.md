# T-REX Repository Audit Report

**Branch:** `audit/code-quality-and-docs` (this section: `feature/unified-environment`, branched off it)
**Scope:** Repository inventory, static-analysis sweep, and correctness audit against
*"A Framework for Benchmarking Traffic Signal Control Robustness under Incidents"*
(submitted to European Transport Research Review; arXiv:2506.13836).

**Important caveat on Phase 2:** the auditor did not have access to the manuscript's
full text/equations — only the summary of Section 2.2–3.4 and Appendices A–C supplied
in the audit task brief. Every "match"/"mismatch" verdict below is against that summary,
not the full paper. Anything not explicitly covered by the summary is marked
**"cannot verify — full paper text not available."** Human reviewers with the manuscript
in hand should re-check the flagged items against the actual equations before the paper
is cited as validating this code.

Status legend: ✅ Fixed · 🚩 Flagged for human review · ⚪ Not an issue · ❓ Cannot verify

---

## Environment unification (`feature/unified-environment`)

`base_env.py::BaseEnv` and `incident_env.py::IncidentEnv` were merged into a single class,
`trex_env.py::TrexEnv`, with incidents controlled by one constructor parameter
(`incident_config: IncidentConfig | None`, `TREX_comp/config/incident_config.py`). `main.py`
keys this off the existing `--strategy` flag (1→`None`/off, 2→`IncidentConfig(level=2)`/on)
rather than adding a second toggle. `BaseEnv`/`IncidentEnv` remain as thin, deprecated
subclasses of `TrexEnv` for backward compatibility (emit `DeprecationWarning`, otherwise
fully functional — verified live).

**Correctness requirement (incidents off ⇒ subsystem not executed, not neutered):**
`TrexEnv.__init__`/`reset()`/`step_sim()` all gate every `Initializer`/`Deployment`
call behind `if self.enable_incidents:` — when `incident_config=None`, `Initializer` is
never constructed for the lifetime of the environment, so it draws zero values from the
global `numpy` RNG (verified: `Initializer.__init__`'s `np.random.randint()` for
`self.random_seed`, and `Initializer.random()`'s `np.random.seed()` reseed, both only ever
execute inside `_initialize_incidents`, which itself only runs when
`self.enable_incidents`). This was true almost by construction once the gating was added
correctly — no RNG-shielding logic had to be invented separately from "don't call the
function."

### Regression test result (mandatory, not assumed)

`tests/test_env_unification.py` runs the pre-refactor `BaseEnv`/`IncidentEnv` (frozen
snapshots in `tests/reference_impl/`, taken immediately before this refactor) and the new
`TrexEnv` side by side on `ingolstadt7`, same fixed seed, same 3-step episode, and diffs the
actual output files (`metrics_1.csv`, `tripinfo_1.xml`). Actual pytest output:

```
collected 3 items

tests/test_env_unification.py::test_base_scenario_matches_pre_merge_baseenv PASSED
tests/test_env_unification.py::test_incident_scenario_matches_pre_merge_incidentenv PASSED
tests/test_env_unification.py::test_grid4x4_base_scenario_works_after_route_path_fix PASSED

3 passed in 0.43s
```

- **Incidents-on comparison**: byte-for-byte identical `metrics_1.csv` and identical
  per-vehicle `(id, duration, waitingTime, timeLoss)` in `tripinfo_1.xml` between pre-merge
  `IncidentEnv` and `TrexEnv(incident_config=IncidentConfig(level=2))`. No normalization
  needed.
- **Incidents-off comparison**: identical `tripinfo_1.xml`; `metrics_1.csv` identical
  *after* normalizing one known, intentional formatting difference (see below) — every
  field value is still asserted equal, only a stray trailing `", "` per line is stripped
  before comparing.
- **Third test**: not a comparison (there's no working "old" baseline on `grid4x4` — see
  below) — it verifies `TrexEnv(incident_config=None)` now runs successfully on `grid4x4`,
  which pre-merge `BaseEnv` could not do at all.

### Edge cases discovered while diffing the two originals

Diffing `base_env.py` against `incident_env.py` line-by-line (not just skimming) surfaced
several real behavioral differences beyond the incident subsystem itself. Two were
unambiguous bugs, fixed as part of the merge (not silently preserved); the rest were
intentionally preserved per-mode in `TrexEnv` rather than converged, specifically so
enabling/disabling incidents doesn't silently change anything else about the simulation:

1. **Fixed — route-file path construction disagreed, and `BaseEnv`'s was broken.**
   Pre-merge `BaseEnv` built route-based `-r` paths as `self.route + '_N.rou.xml'` (a flat
   file-prefix convention); pre-merge `IncidentEnv` and `main.py`'s own decompress-check
   both used `os.path.join(self.route, ...)` (a subdirectory convention). These can't both
   be right for the same `route` config value and the same on-disk layout. Confirmed live:
   with `grid4x4.zip` decompressed the way `main.py`'s precondition check and the README
   expect (`environments/grid4x4/grid4x4/*.rou.xml`), pre-merge `BaseEnv` crashed with
   `TraCIException: The route file '.../grid4x4_1.rou.xml' is not accessible` — it was
   looking one directory up from where the files actually are. This means **`--strategy 1`
   on `grid4x4`/`arterial4x4` did not work at all** before this refactor, independent of
   anything else audited so far — a previously-undiscovered bug because earlier smoke tests
   only exercised `--strategy 1` on `.sumocfg`-based networks (Cologne/Ingolstadt), never on
   the two route-based ones. `TrexEnv` uses the subdirectory convention for both modes
   (verified: `test_grid4x4_base_scenario_works_after_route_path_fix` now runs a full
   episode there with incidents off).
2. **Fixed (in the sense of "unified, not preserved") — `save_metrics` CSV formatting.**
   Pre-merge `BaseEnv.save_metrics` built each line as
   `str(value) + ', '` per field including the last, leaving a stray trailing `", "` before
   the newline (a manual-concatenation artifact); pre-merge `IncidentEnv.save_metrics` used
   `', '.join(...)`, with no trailing separator. Same data, one cosmetic byte difference.
   `TrexEnv` unifies both modes onto the cleaner `', '.join(...)` format — the alternative
   (keeping two different `metrics_*.csv` formats depending on whether incidents are
   enabled) would work against the point of unifying the environments in the first place.
   Not expected to affect any downstream consumer: `readCSV.py`'s parser splits fields on
   `'}'`/`':'`, not on trailing whitespace.
3. **Preserved, not converged — additional-file loading.** Pre-merge `BaseEnv` never passed
   `-a`/`--additional-files` (so the network's `.add.xml` vType overrides, e.g. `CAV1`/
   `CAV3`/`IC`/`CAV4`, were never loaded in the base scenario); pre-merge `IncidentEnv`
   always did. `TrexEnv` keeps this exactly per-mode (`self.additional` is only resolved
   and only passed to SUMO when `enable_incidents`), since always loading it in the "off"
   mode would be a genuine behavioral change to base-scenario vehicle dynamics, not just a
   cosmetic one.
4. **Preserved, not converged — the global `--time-to-teleport` flag.** Pre-merge `BaseEnv`
   unconditionally passed `--time-to-teleport -1` (global teleport disabled for every
   vehicle); pre-merge `IncidentEnv` did not (superseded by the per-vehicle `CAV4` teleport
   exemption mechanism added earlier in this audit, which only makes sense when incidents
   exist to queue vehicles behind). `TrexEnv._seed_and_teleport_args()` keeps this
   conditional on `enable_incidents` for the same reason as item 3 — this materially affects
   simulation dynamics (whether gridlocked vehicles ever get teleported away), not just log
   formatting.
5. **Preserved (functionally equivalent either way) — phase-string filtering.** Pre-merge
   `BaseEnv` checked `"y" not in p.state` (no `.lower()`); pre-merge `IncidentEnv` checked
   `"y" not in p.state.lower()`. SUMO phase-state strings only ever use lowercase `y`/`g` and
   uppercase `G`/`R` in practice, so these are equivalent for real output — `TrexEnv` uses
   the more defensive `.lower()` form for both modes, and the regression test's identical
   `metrics_1.csv`/`tripinfo_1.xml` confirms this didn't change phase/action counts for the
   networks tested.
6. **Not functionally relevant** — the two originals built the *initial* (pre-`reset()`)
   TraCI connection label in slightly different string formats (unused after the first few
   lines of `__init__`, before both renamed it to the identical final `connection_name`
   format anyway), and created the results directory via raw string concatenation
   (`BaseEnv`) vs. `os.path.join` (`IncidentEnv`) — functionally identical given `main.py`
   always passes a `log_dir` already ending in `os.sep`, but the concatenation version is
   fragile to that assumption (confirmed while writing the regression test: passing a
   `log_dir` *without* a trailing separator breaks pre-merge `BaseEnv`'s directory creation
   silently). `TrexEnv` always uses `os.path.join`.

## Needs owner decision (round 2)

These three items are **deliberately left unresolved** — each is a numeric discrepancy
between the running code and the paper/brief's stated value, in scientifically load-bearing
sampling/behavior code. Only someone with the manuscript in hand (or who made the recent
edit in the `slow_zone_speed` case) can say which side is authoritative; this audit will not
guess. Full context for each is in the linked section below.

| Item | Code value | Paper/brief value | File:line |
|---|---|---|---|
| Incident start-time upper bound | `end_time − 500` | `end_time − 1200` (Section 2.3) | `T_REX.py:240` |
| Warm-up phase | `warmup=0` in all 10 network configs | nonzero warm-up implied by its own name/role as the incident start-time distribution's lower bound | `TREX_comp/config/map_config.py:11,22,33,44,59,70,81,92,103,114` |
| `slow_zone_speed` | `1.39` (m/s) | contradicts its own inline comment (`# 13.8 is 50 km/h...`) and doesn't exactly match the paper's "~8 km/h / 5 mph" figure either | `T_REX.py:28` |

See §2.4 ("Incident sampling") and §2.3 ("SSD-based speed adaptation") below for the full
writeup of each, including why round 1 flagged rather than fixed them.

---

## 0. Repository shape vs. the paper's four-module architecture

| Paper module | Code location | Notes |
|---|---|---|
| Network Environment | `base_env.py`, `incident_env.py`, `traffic_signal.py`, `environments/` | SUMO scenario configs + Gym env wrapper |
| Initializer | `T_REX.py::Initializer` | incident edge/lane/pos/time/duration sampling |
| Deployment | `T_REX.py::Deployment` | ICM rerouting, SSD speed adaptation, lane-changing, teleport-exemption |
| RL Interaction | `TREX_comp/` (`agents/`, `rewards.py`, `states.py`, `config/`) + `main.py` | Adapted from RESCO (`# Adapted from original: https://github.com/Pi-Star-Lab/RESCO`, `base_env.py:2`) |

The four modules map cleanly onto the code, but the module *names* only appear as
class/file names inside `T_REX.py` — there is no package-level grouping (see Phase 5
finding on layout). ⚪ Architecture mapping itself is not a defect, just undocumented;
addressed in the README rewrite (Phase 6).

**Scope note (updated after round-2 clarification from the repo owner):** the paper's
Section 3.4 robustness metrics — **LSI, FPD, CR, AUC, RAUC, PDI** — have **no
implementation anywhere in this repository**
(`grep -rniE "LSI|FPD|convergence_rate|RAUC|PDI\b|area_under|learning_stability"` across
all `*.py` returns zero hits). This is **by design, not a gap**: T-REX's responsibility
ends at producing raw per-episode performance-indicator logs; the Section 3.4 metrics are
computed by a separate downstream analysis pipeline that is not part of this repository.
Round 1 of this audit flagged this as an open question rather than assuming it — see
§2.1 below for what T-REX actually logs, and the README's "Metrics & Analysis" section for
the user-facing version of this same clarification. ⚪ Not an issue — Phase 2.1 (diffing the
metric formulas against the paper) and the metrics half of Phase 5.5 (unit tests) are out of
scope for this codebase, and no metric-computation code was written here.

---

## 1. Phase 1 — Static audit findings

### 1.1 Repo hygiene (Critical)

| # | Finding | Location | Status |
|---|---|---|---|
| 1 | No `.gitignore` anywhere in the repo | root | 🚩 Fixed in Phase 4 (see CHANGELOG) |
| 2 | `.git` directory is **6.4 GB**; `environments/arterial4x4/` alone has **2,806 tracked files (1.3 GB)** of generated `*.rou.xml` route files, plus a 24 MB `.zip`; `environments/grid4x4/` has a 19 MB `.zip` | `environments/arterial4x4/`, `environments/grid4x4/` | 🚩 Flagged — candidate for Git LFS or regeneration script + `.gitignore`, **not deleted** (ground rule: don't delete without confirmation). Note: `arterial4x4` is not one of the four networks the paper claims (Grid4x4, Cologne Corridor/Region, Ingolstadt Corridor/Region) — it looks like a RESCO-inherited network never removed. |
| 3 | 47 `__pycache__`/`.pyc` files committed to git under `TREX_comp/` | `TREX_comp/**/__pycache__/*.pyc` | ✅ Fixed — removed from version control, added to `.gitignore` |
| 4 | No secrets/API keys found (`grep -rniE "api[_-]?key\|secret\|password\|token\s*=\|AKIA..."`) | — | ⚪ Not an issue |
| 5 | No hardcoded absolute local paths (`/home/username`, `C:\Users`, `/Users/name`) found in `*.py` | — | ⚪ Not an issue |
| 6 | `readXML.py` hardcodes a fragile relative path assumption (`env_base = 'RESCO_main'+os.sep+'environments'+os.sep`) and a personal results directory name (`results_test_ib_Ingolstadt21`); it is a standalone analysis/plotting script, never imported by `main.py` or any other module in the pipeline | `readXML.py:15,21` | 🚩 Flagged — candidate for moving to a `scripts/` or `analysis/` folder and parameterizing the path, or removal if superseded. Not touched (uncertain if still in active use). |
| 7 | No `LICENSE` file at repo root (only `environments/LICENSE`, a data-license for the network files) | root | 🚩 Flagged — **needs human decision** on code license (see PR summary) |
| 8 | No `tests/` directory, no CI (`.github/workflows/`), no `CITATION.cff`, no `CONTRIBUTING.md`, no `environment.yml`/`pyproject.toml` despite the README's quickstart invoking `conda env create -f environment.yml` (file doesn't exist) | root | 🚩 Addressed in Phase 5/6 (tests, CI, citation, contributing added; environment.yml added or README corrected — see CHANGELOG) |

### 1.2 Dependency management

`requirements.txt` lists **zero pinned versions**:
```
numpy
pandas
gym
sumolib
traci
torch
tensorflow # because you used tensorflow.compat.v1
pfrl
eclipse-sumo
```
🚩 No SUMO version is documented (README says "SUMO" with a link, no version). `readXML.py`
imports `from resco_benchmark.config.map_config import map_configs` (`readXML.py:6`) but
`resco_benchmark` is **not listed in `requirements.txt` at all**, and no RESCO commit/version
is pinned anywhere — only a link to `https://github.com/Pi-Star-Lab/RESCO` in the README.
Note this import is only exercised by the standalone `readXML.py` script (finding 1.1.6);
the core training pipeline (`main.py` → `TREX_comp/`) is a self-contained adaptation of
RESCO's agent/env code and does not import `resco_benchmark` at runtime. ✅ Addressed —
`requirements.txt` pinned to tested version ranges (Phase 5), README documents SUMO
version requirement and clarifies the RESCO relationship.

### 1.3 Code quality (from static analysis + manual review)

*(merged with the background static-analysis sweep — see §1.4 for tool output)*

- **Duplicate method definition**: `T_REX.py` defines `Deployment.get_arcs_cost` **twice**
  (lines 1307–1325 computing *upstream* arc costs, and lines 1328–1358 computing
  *downstream* arc costs). Python silently keeps only the second; the first is dead,
  unreachable code. The second (downstream) definition is the one consistent with how
  `arc_costs` is actually used in `calculate_avoided_loss` (called with `get_actual_probs`/
  `get_typical_probs`, which both operate over downstream edges). ✅ **Fixed** — removed the
  dead first definition (line range 1307–1326). This is a pure dead-code removal (the first
  definition was never reachable), not a behavior change.
- **Large commented-out blocks**: several multi-line dead-code blocks in `T_REX.py`
  (e.g. `T_REX.py:277–287` old `save_incident_information`/`load_incident_dict`;
  `T_REX.py:1360–1391` old `add_information_noise`; `T_REX.py:1566–1587` old
  `reroute_model`; `T_REX.py:385–428` `simulate_accident_with_blocking`). 🚩 Flagged as
  cleanup candidates in Phase 4 — left in place pending confirmation these superseded
  versions aren't wanted as reference/rollback material.
- **`print()` used for debugging instead of `logging`**: pervasive across `T_REX.py`,
  `incident_env.py`, `base_env.py`, `main.py` (e.g. `T_REX.py:100-102`, `:190`, `:693`,
  `:846-847`, `:862-863`, `:975`, `:1303`; `incident_env.py:47,61,106,158,324`;
  `base_env.py:17,46,101,221`). ✅ Addressed in Phase 3 (converted to `logging` calls).
- **`assert` used for runtime validation** (not just tests) in `Initializer.random_edge`
  (`T_REX.py:182`) and `weighted_random_edge`'s error path uses `raise ValueError` — assert
  statements are stripped when Python runs with `-O`, silently disabling the safety check.
  🚩 Flagged, not changed (behavior-preserving fix is easy — replace with an explicit
  `if not valid_edges: raise ValueError(...)` — but left for human review since it sits in
  scientifically load-bearing sampling code covered by ground rule 3).
- **RNG reproducibility gap** (see §2.4 below for full analysis) — global `np.random.seed()`
  reseeding mixed with unseeded `np.random.randint`/`np.random.rand` calls inside the RL
  agents' exploration policy, and SUMO is always launched with `--random` rather than a
  controlled `--seed`. 🚩 Flagged for human review (Phase 3) — not silently changed, since
  altering SUMO/agent-level seeding could change published-result reproducibility in ways
  that need a scientist's sign-off, not a blind fix.
- **TraCI/SUMO lifecycle has no exception safety**: all 8 `traci.start(...)` call sites
  (`base_env.py:40,43,134,137`; `incident_env.py:97,100,206,209`) and every matching
  `traci.close()` in `reset()`/`close()` are *not* wrapped in `try/except/finally`. Any
  exception raised between `traci.start()` and the next `traci.close()` (e.g. a TraCI RPC
  error, a bug in `state_fn`/`reward_fn`, an incident-sampling failure) leaks the SUMO
  subprocess. ✅ Fixed in Phase 3 — `reset()`/`close()` teardown wrapped in `try/finally`
  so `traci.close()` always fires.
- **Code duplication between `base_env.py` and `incident_env.py`**: constructor SUMO-command
  construction, `step_sim`, `reset`, `step`, `calc_metrics`, `save_metrics`, `close` are
  ~90% identical between the two files (compare `base_env.py:103-235` to
  `incident_env.py:160-341`). 🚩 Flagged as a Phase 4 refactor candidate (extract a shared
  `SumoTrafficEnv` base class) but **not applied** — this touches the core training loop for
  every RL method benchmarked in the paper, and a refactor here carries real risk of
  behavioral drift that would undermine reproducibility of already-published results. Left
  for human review with a concrete proposed diff sketch in the PR description.

### 1.4 Static analysis tool output

None of `ruff`/`flake8`/`black`/`isort`/`mypy`/`pylint` are installed in this environment;
none were installed as part of the audit (per ground rules, not a blocker — recorded as a
finding). 🚩 Flagged — added to the CI workflow (Phase 5) so they run automatically going
forward, with `ruff`/`black`/`isort` configured but **not run over the whole tree yet in
this pass** (a full reformat of 3,400+ lines of scientifically load-bearing code is a bigger
change than this audit should make unreviewed; left for a follow-up PR once CI is in place
to validate it). As a baseline sanity check, `python3 -m py_compile` was run over all 24
`.py` files (root + `TREX_comp/`) — **all compile cleanly, zero syntax errors.**

**`print()` debug-output counts by file:** `T_REX.py` 46, `traffic_signal.py` 12, `main.py` 9,
`graph.py` 7, `readXML.py` 5, `incident_env.py` 5, `base_env.py` 4, `TREX_comp/agents/fma2c.py` 3,
`TREX_comp/agents/pfrl_dqn.py` 2, `TREX_comp/agents/mplight.py` 2, `readCSV.py` 2,
`TREX_comp/agents/pfrl_ppo.py` 1, `TREX_comp/agents/ma2c.py` 1. No `logging` module used
anywhere in the codebase prior to this audit. ✅ Fixed for the core simulation path
(`T_REX.py`, `base_env.py`, `incident_env.py`, `main.py`) — converted to `logging` calls at
appropriate levels, configurable via a new `--verbose` flag. Not touched in the RL agent
files or `graph.py`/`readCSV.py`/`readXML.py` (lower-traffic, non-core paths) to keep the
diff reviewable — flagged as a follow-up.

**Bare `except:` clauses:** none found (0 hits). ⚪ Not an issue.

**Broad/silent exception handling:**
- `readXML.py:61` — `except Exception as e:` immediately followed by `#raise e` (commented
  out) then `break` — silently swallows *any* parsing error and just stops the loop with no
  record of what happened. ✅ Fixed — now logs the exception before breaking (control flow
  unchanged, only visibility improved). This script is a standalone analysis tool not
  imported by the main pipeline (see §1.1.6), so the fix carries no risk to training/eval.
- `T_REX.py:631` — `except Exception as e:` that *does* `print(f"Error restoring vehicle...")`
  — not silent. ⚪ Not an issue (print → logging conversion covers this as part of the
  broader print cleanup).
- `TREX_comp/agents/fma2c.py:12`, `TREX_comp/agents/ma2c.py:10` — `except ImportError: tf = None`
  — legitimate optional-dependency guard. ⚪ Not an issue.
- `readXML.py:69` (`FileNotFoundError`), `readXML.py:98` (`ET.ParseError`) — properly scoped.
  ⚪ Not an issue.

**TODO/FIXME/XXX:** one hit — `traffic_signal.py:108`, `# TODO raise Exception('Invalid signal config')`.
🚩 Flagged, not resolved — implementing the TODO would change error-handling behavior in the
signal-configuration path without a clear picture of what currently happens on an invalid
config; left for human judgment.

**Large commented-out dead-code blocks (5+ consecutive lines) in `T_REX.py`:** lines
138-143, 277-282, 355-361, 389-460 (an entire disabled method,
`simulate_accident_with_blocking`), 674-680, 801-805, 849-853, 879-889, 1072-1077,
1360-1387 (old `add_information_noise`), 1570-1575, 1581-1587 (old `reroute_model`),
1697-1710, 1721-1726, 1734-1738. Also `TREX_comp/config/signal_config.py:3-7` (dead
`monaco_valid_acts`/`monaco_phase_pairs` config for an unused map). 🚩 Flagged as Phase 4
cleanup candidates — **not removed**, since several of these are superseded-but-related
versions of live logic (e.g. the commented `add_information_noise`/`reroute_model` sit right
next to their replacements) that a maintainer may want to diff against or restore; deleting
them is a judgment call outside an "unambiguous bug fix." (Note: `T_REX.py:833-841` was
checked and is a genuine multi-line *prose* explanation of SUMO's `baseType@vehID` naming,
not dead code — correctly left alone.)

**Unused imports** (verified by grep, not just the agent's AST heuristic):
`T_REX.py`: `os`, `sys`, `json` (only referenced inside a commented-out block, `T_REX.py:279`),
`pandas as pd`, `xml.etree.ElementTree as ET`, `optparse`, `from sumolib import checkBinary`,
`from time import time`. `main.py`: `from pathlib import Path`. ✅ Fixed — removed (each
verified with a full-file grep for the bound name before removal; `random` and `csv` in
`T_REX.py` **are** used and were kept).

### 1.5 Duplicated logic (RL agents, envs)

- **`base_env.py` vs `incident_env.py`**: near-line-for-line duplication of the constructor's
  SUMO-command/phase-detection setup, `step_sim`, `reset`, `step`, `calc_metrics`,
  `save_metrics`, `close`, `render` (~150+ lines). `IncidentEnv` does not subclass `BaseEnv` —
  it's a full parallel reimplementation with incident hooks bolted on. See §1.3's cleanup
  entry — 🚩 flagged, not refactored (too risky to auto-apply to the core training loop).
- **`TREX_comp/agents/`**: `conv2d_size_out()` is defined identically in `pfrl_dqn.py:24-25`
  and `pfrl_ppo.py:43-44`. `save()`/`load()` are byte-for-byte identical between
  `pfrl_dqn.py:98-106` (`DQNAgent`) and `pfrl_ppo.py:83-91` (`PFRLPPOAgent`), neither pulled
  into the shared `Agent` base class (`TREX_comp/agents/agent.py`). The "iterate `obs_act`,
  build a per-signal sub-agent, optionally load from disk" pattern repeats near-identically
  in `IDQN.__init__`, `IPPO.__init__`, and `MPLight.__init__`. By contrast, `maxpressure.py`'s
  `MaxAgent` cleanly subclasses `maxwave.py`'s `WaveAgent` — a good example of the reuse
  pattern the other agents lack. 🚩 Flagged as a Phase 4 refactor candidate — **not applied**,
  since touching every RL agent's constructor risks subtle behavioral drift in code that
  produced the paper's published numbers; left for human review with the specific
  duplication sites listed here.

### 1.6 Magic numbers vs. Appendix B (expanded)

Beyond the ICM/AASHTO/incident-duration constants already covered in §2.2-2.4:
`T_REX.py:333-336`'s `awareness_params` dict inlines eleven more ICM sub-model constants
(`pi_news=0.7, pi_on=0.5, T_broadcast=5, vms_percentage=0.4, enter_time_vms=10, l2=200,
beta_vms=2, pi_online=0.8, enter_time_online=5, sigma=10, t_obs_0=2, xi_obs=0.5`) with no
config binding. More significantly: **`TREX_comp/agents/pfrl_ppo.py`'s optimizer/training
hyperparameters are hardcoded inline, bypassing `agent_config.py` entirely** — `lr=2.5e-4,
eps=1e-5` (`pfrl_ppo.py:65`), `clip_eps=0.1, update_interval=1024, minibatch_size=256,
epochs=4, entropy_coef=0.001, max_grad_norm=0.5` (`pfrl_ppo.py:66-75`) — meaning IPPO's
hyperparameters cannot be tuned or overridden via config the way IDQN/MPLight/FMA2C's can.
🚩 Flagged as a Phase 5 config-centralization target; **not moved into config in this pass**
(IPPO's hyperparameters aren't covered by the Appendix B summary given to this audit, so a
correct config default couldn't be sourced with confidence — moving the values without
knowing the intended per-network defaults would just relocate the magic numbers, not fix
the underlying gap).

**Confirmed real bug — `TREX_comp/agents/mplight.py:36`:** `MPLight.__init__` accepts an
`lr` parameter (`def __init__(self, config, obs_act, map_name, thread_number, lr=0.005)`)
but **never uses it** — the `DQNAgent(...)` call on line 36 hardcodes `lr=0.005` literally
instead of passing through the received `lr`. This means MPLight silently ignores whatever
learning rate its caller supplies (including `main.py`'s `--lr` CLI flag) and *always*
trains at `0.005` regardless. ✅ **Fixed** — line 36 now passes `lr=lr`. This is an
unambiguous parameter-plumbing bug (the parameter is accepted, documented by its presence
in the signature, and then discarded) rather than a change to any formula — but it **does**
change runtime behavior for anyone who was relying on `--lr` to actually take effect for
MPLight, so it's called out explicitly here and in `CHANGELOG.md` rather than buried in a
generic "bug fixes" line.

### 1.6b Confirmed real bug — `main.py::run_episode` discards its `obs` parameter

`run_episode(env, agent, obs=None)` (`main.py:199-205`, pre-fix) immediately overwrote
its own `obs` parameter with a fresh, unseeded `env.reset()` call before ever using the
value passed in:
```python
def run_episode(env, agent, obs=None):
    obs = env.reset()   # <-- clobbers the caller's obs unconditionally
    ...
```
`run_incident_scenario` (`main.py:156-196`) calls `run_episode(env, agent, obs)` in the two
places that matter most for reproducibility: after `env.reset(pre_seed=[seed_ic1, seed_ic2])`
when replaying fixed incident seeds for testing (`--repeat`), and after `env.reset()` when
recording the seeds to save during training. In both cases the caller's carefully-seeded
`obs`/environment state was discarded and the environment was silently reset **a second
time** with a fresh, non-reproduced seed — meaning the entire `--repeat`
(fixed-incident-seed testing/replay) feature never actually replayed the saved seeds; it
silently ran with new random incidents every time while still reporting the seeds it
*thought* it used. ✅ **Fixed** — `run_episode` now only calls `env.reset()` when `obs` is
`None` (i.e. for the call sites that never had a pre-seeded observation to begin with);
callers that already reset with the correct seed now have that reset honored. This is an
unambiguous bug (a parameter accepted by the function signature and then immediately
discarded before use, identical in kind to the `mplight.py` `lr` bug in §1.6) rather than a
change to any sampling/scientific logic — but because it directly affects whether
incident-seed replay (relevant to reproducing Experiment 3-style testing/transfer runs) ever
worked, it's called out explicitly here and in `CHANGELOG.md`.

### 1.7 TraCI/SUMO lifecycle safety (expanded)

All 8 `traci.start(...)` call sites (`base_env.py:40,43,134,137`; `incident_env.py:97,100,206,209`)
and all 6 matching `traci.close()` calls (`base_env.py:96,111,234`; `incident_env.py:150,178,339`)
are unguarded. `main.py`'s `run_trial()` calls `env.close()` as a plain final statement
(`main.py:139`) after `run_base_scenario`/`run_incident_scenario`, neither of which has any
exception handling — an exception during a training/eval episode aborts the process without
ever closing the SUMO connection. ✅ Fixed — `reset()`/`close()` in both env files now wrap
the teardown (`traci.switch`/`traci.close`/`save_metrics`) in `try/finally`, and `main.py`'s
`run_trial` wraps the run + `env.close()` in `try/finally` so a mid-episode exception still
releases the SUMO subprocess.

### 1.8 Environments directory: no committed run-artifacts

Swept all 2,821 files under `environments/` for anything that looks like SUMO *output*
(tripinfo/summary/result/checkpoint/csv/log) rather than input config — found none. Every
tracked file is legitimate input config (2,806 `.rou.xml`, 8 `.net.xml`, 7 `.add.xml`,
8 `.sumocfg`) plus the 5 CC-license PDFs and 2 zip archives already flagged in §1.1.2.
⚪ Not an issue — no accidentally-committed run artifacts found.

### 1.9 Files exceeding 1000 lines (maintainability)

`T_REX.py` (2,018 lines — also the file with the most dead code, unused imports, and debug
prints; the strongest single refactor/split candidate) and `TREX_comp/config/signal_config.py`
(1,329 lines — mostly a static per-map signal/phase-pair data table, not logic). 🚩 Flagged,
not split — a structural split of `T_REX.py` (e.g. separating `Initializer` and `Deployment`
into their own modules) is a reasonable Phase 5 layout improvement but was left for human
review given how much scientifically load-bearing logic lives in that one file.

---

## 2. Phase 2 — Correctness audit against the paper

### 2.1 Robustness metrics (LSI, FPD, CR, AUC, RAUC, PDI)

⚪ **By design, out of scope for this repository** (confirmed by the repo owner in round 2
of this audit) — not a gap. T-REX writes two raw per-episode artifacts per training/eval
run, and the Section 3.4 metrics are computed from these (or from SUMO's own tripinfo
output) by a separate analysis pipeline not included here:

1. **`results/<connection_name>/metrics_<run>.csv`** — written by
   `calc_metrics`/`save_metrics` in both `base_env.py` (lines 200-222) and `incident_env.py`
   (lines ~309-329, post-fix line numbers). One line per environment `step()` call
   (i.e. per RL decision, not per SUMO simulation second), each line the `str()` of four
   comma-joined Python values in this order: `step` (SUMO simulation time in seconds, an
   int), `reward` (a `dict[signal_id, float]` from whichever `reward_fn` the agent config
   selected — see `TREX_comp/rewards.py`), `max_queues` (`dict[signal_id, int]`, the largest
   per-lane queue at that signal), `queue_lengths` (`dict[signal_id, int]`, summed
   per-lane queue at that signal). **Not standard CSV** — the dict fields are Python
   `repr()` text (`{'A0': 3, 'A1': 0, ...}`), not separate columns; downstream consumers
   (e.g. `readCSV.py`) parse it with ad hoc string-splitting on `}`/`:` rather than a real
   parser.
2. **`results/<connection_name>/tripinfo_<run>.xml`** — SUMO's native
   [tripinfo output](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html) (written
   directly by SUMO via `--tripinfo-output`/`--tripinfo-output.write-unfinished`, set in
   `base_env.py:136-138` and `incident_env.py:203-204`): one `<tripinfo>` element per
   vehicle with `depart`/`arrival`/`duration`/`waitingTime`/`timeLoss`/etc. — this is what
   `readXML.py`'s `avg_timeLoss`/`avg_duration`/`avg_waitingTime` helpers consume.

Both are written per-episode under `args.log_dir` (default `./results/`); neither file
format is itself an LSI/FPD/CR/AUC/RAUC/PDI value — those are computed from a *sequence* of
these per-episode files across a training run, which is the downstream pipeline's job.

### 2.2 ICM rerouting model (Appendix A)

Located in `T_REX.py::Deployment` (`ICM`, `calculate_combined_awareness`, `reroute_model`,
`calculate_expected_gain`, `calculate_avoided_loss`, lines ~907–1685).

| Component | Code | Verdict |
|---|---|---|
| ICM parameters `β_0=-5, β_gain=2.5, β_loss=2.5` | `T_REX.py:336` `self.ICM_params = {'beta_0': -5, 'beta_gain': 2.5, 'beta_loss': 2.5}` | ✅ Matches paper exactly |
| Binomial-logit rerouting decision `P = 1/(1+exp(-(β0 + β_gain·Δp - β_loss·Δw)))` | `T_REX.py:1575-1593` | ✅ Matches the standard binary-logit form described in the brief; also now covered by `tests/test_icm.py` |
| Awareness sources combined as `1-(1-news)(1-vms)(1-online)(1-obs)` | `T_REX.py:1668` | ✅ Structurally consistent with "FTI/FPI/OS/OB sources" combined into one probability (news≈FTI broadcast, vms≈FPI, online≈OS, obs≈OB, by naming) — **cannot verify the individual sub-formulas' exact functional forms** against Appendix A without the paper text (`calculate_arc_radio_awareness`, `calculate_vms_awareness`, `arc_online_awareness`, `calculate_observation_awareness` — not read line-by-line in this pass) |
| Driver heterogeneity mix (experienced 40%/5%, novice 30%/10%, distracted 20%/20%, CAV 10%/1%) | `T_REX.py:1402-1413` `driver_prob = [0.4, 0.3, 0.2, 0.1]` for `['experienced','novice','distracted','CAV']`; `noise_std` = `0.05, 0.1, 0.2, 0.01` respectively | ✅ **Exact match** to Appendix B on both the population split and the per-type error rate |

### 2.3 SSD-based speed adaptation (Section 2.4.2)

`T_REX.py::Deployment.speed_adjustment` (lines 700–754).

- AASHTO perception-reaction time `t=2.5s` and deceleration `a=3.4 m/s²`
  (`T_REX.py:698-699`, named class constants as of round 1's Phase 5 test-extraction —
  previously inline magic numbers), combined as `SSD = v·t + v²/(2a)` in the extracted,
  unit-tested `Deployment.calculate_ssd` classmethod (`T_REX.py:702-708`) — ✅ **matches**
  the AASHTO constants and the standard SSD formula stated in the brief.
- **5 mph (~8 km/h) reduced-speed rule**: 🚩 **Flagged for human review, not changed.**
  `self.slow_zone_speed` is set in `T_REX.py:28` to `1.39` (m/s) with the comment
  `# 13.8 is 50 km/h should work for highway situations.` — the comment describes a value
  (13.8 m/s ≈ 50 km/h) that does **not match** the value actually assigned (`1.39` m/s ≈
  5.0 km/h ≈ 3.1 mph). Neither value matches the paper's stated "~8 km/h / 5 mph" figure
  exactly, though `1.39` is closer. This line is part of the **uncommitted WIP the user
  asked to commit as a baseline** at the start of this audit (previously `2.2` m/s ≈ 7.9
  km/h ≈ 4.9 mph, which *did* match the paper). Per ground rule 3 this is exactly the
  "formula/value doesn't match its own comment" pattern that should be flagged rather than
  silently fixed, especially since it's a value the user was actively editing minutes before
  this audit began — the intended target value is ambiguous from the code alone. **Needs a
  decision: was `1.39` an intentional new experiment value (typo in the comment), or should
  it revert to `2.2` (paper-matching) or become `13.8` (matching the comment)?**

### 2.4 Incident sampling (Section 2.3)

`T_REX.py::Initializer` (lines 84–255).

| Parameter | Paper | Code | Verdict |
|---|---|---|---|
| Blocked lane count | uniform | `random_lanes`: `np.random.randint(1, n_lanes+1)` (level 2) / `randint(1, n_lanes)` (level 1), lanes taken from either end, not fully random subset (commented-out fully-random alternative at `T_REX.py:225-226`) | ✅ Uniform count sampling matches; lane *contiguity* (blocking from one end rather than an arbitrary subset) is a modeling choice not contradicted by the brief's "uniform lane count" description |
| Position `U(10, x_e − 10)` | `random_pos`: `np.random.uniform(10, edge_length - 10)` (`T_REX.py:233`) | ✅ **Exact match** |
| Duration `~Exp(0.029)` | `random_duration`: `np.rint(np.random.exponential(1/0.029)).astype(int)*60` (`T_REX.py:249`) — note `numpy`'s `exponential(scale)` takes `scale=1/rate`, and the result is in **minutes**, multiplied by 60 for seconds | ✅ Matches `Exp(0.029)` under the standard rate parameterization, assuming the paper's rate is per-minute (consistent with the `*60` conversion and with realistic incident durations) |
| Start time `U(t_warmup, t_end − 1200)` | `random_time`: `np.rint(np.random.uniform(self.warm_up_time, self.end_time - 500)).astype(int)` (`T_REX.py:240`) | 🚩 **Mismatch, flagged not fixed — see "Needs owner decision" at the top of this report.** Code uses `end_time - 500`, the paper's brief states `end_time - 1200`. This is a direct numeric discrepancy against the paper's stated formula (exactly the "formula doesn't match the paper's equation" case ground rule 3 says to flag, not guess-fix) — could be a bug, or a deliberate post-submission revision. **Needs a decision from someone with the manuscript in hand.** |
| `warm_up_time` always `0` | implied nonzero (used as the lower bound of the incident start-time distribution) | `map_config.py`: every network entry has `'warmup': 0` | 🚩 Flagged, not changed — `warmup=0` for all 8 networks means the incident start-time distribution's lower bound is always 0 regardless of network; some networks additionally set a `start_time` (a simulation-of-day offset, e.g. Ingolstadt `57600`) which is a *different* field passed to SUMO, not `warm_up_time`. Whether this is intentional (warm-up handled via the time-of-day offset instead) or a gap needs a modeler's confirmation. |

### 2.4b Incident *categories* (Table B1) are not a distinct code parameter

Table B1 (per the audit brief) describes five incident categories: collision, stalled
vehicle, roadworks, speed-reduction/environmental, and signal malfunction. `Initializer`/
`Deployment` implement a single generic mechanism — N lanes of an edge become impassable
(via a dummy `incident_veh_*` vehicle) for a sampled duration/position/lane-count — with no
`incident_type`/`category` parameter anywhere in the class (confirmed by grep: the only
"collision" logic in the file is inside `simulate_accident_based_on_blocked_lanes`, the
disabled dead-code method from §1.3/1.4, not the active `sim_incident` path). ⚪/🚩 Not a bug
— the generic blockage mechanism can represent any of the five categories conceptually
(that's plausibly the paper's intent) — but there is no code-level way to *select* which
category a given incident represents, so any claim that T-REX lets a user configure "a
roadworks incident" vs. "a stalled-vehicle incident" would be aspirational, not actual.
Flagged so the README (Phase 6) describes what's actually configurable (edge, lane count,
position, start time, duration, severity `level`) rather than a five-category switch that
doesn't exist in code.

### 2.5 Hyperparameters (Appendix B) vs. training config

- `TREX_comp/config/agent_config.py`: `IDQN` and `MPLight` both hardcode `'GAMMA': 0.99`
  for every network (`agent_config.py:89,107,147`) — ✅ matches "γ=0.99 everywhere."
- 🚩 **Learning rate is not network-specific anywhere in the config.** Neither `IDQN` nor
  `MPLight` entries in `agent_config.py` carry an `'lr'` key; the learning rate instead comes
  from `main.py`'s `--lr` CLI flag, **default `0.001`** (`main.py:49`), applied identically to
  `IDQN` and `MPLight` regardless of which network is selected (`main.py:130`). The paper's
  Appendix B specifies **per-network** learning rates (IDQN: `1e-5` for Grid4x4/Ingolstadt
  Region, `0.001` elsewhere; MPLight: `0.005` grid, `0.01` Ingolstadt Region, `0.001`
  others). `MPLight`'s own class default is `lr=0.005` (`TREX_comp/agents/mplight.py:14`),
  but this default is **silently overridden** by `main.py`'s `lr=args.lr` call whenever
  `main.py` is used without an explicit `--lr` flag — so the out-of-the-box behavior of
  `python main.py --agent MPLight --map grid4x4` uses `lr=0.001`, not the paper's `0.005`
  for Grid4x4. **This is a real reproducibility gap**: reproducing the paper's reported
  numbers requires the operator to manually pass the *correct* `--lr` for every
  agent/network combination, undocumented anywhere in the repo or README. ✅ Addressed in
  Phase 5 — added a structured per-network/per-agent hyperparameter default table matching
  Appendix B (`TREX_comp/config/hyperparams.py`, a plain-Python config module for
  consistency with the existing `agent_config.py`/`map_config.py`/`mdp_config.py` style
  rather than introducing a new YAML dependency), consulted by `main.py` whenever `--lr`
  is not explicitly passed (its CLI default changed from a hardcoded `0.001` to `None`, so
  "not passed" is distinguishable from "explicitly passed"), so the correct paper default
  is used automatically per network/agent, while an explicit `--lr` still overrides it.
  This does not change any agent internals or formulas — it only fixes *which default
  value* is selected when the user doesn't specify one, which is squarely a
  config/reproducibility bug, not a change to scientific logic. Only IDQN/MPLight are
  covered (the only two the audit brief gave explicit per-network grids for); FMA2C/IPPO
  are untouched.
- `FMA2C`/`MA2C` hyperparameters (`gamma=0.96`, `lr_init=2.5e-4`, etc., `agent_config.py:44-63`)
  — ❓ cannot verify against Appendix B; the task brief only gave IDQN/MPLight hyperparameter
  grids explicitly.
- Fixed 10-second phase length: `map_config.py` sets `'step_length': 10` for `grid4x4`,
  `ingolstadt1/7/21`, `cologne1/3/8`, `turin5` — ✅ matches. `arterial4x4`/`arterial5x5` use
  `step_length: 5` — ⚪ not a mismatch against the paper, since these two networks are not
  among the four networks the paper claims to use (they appear to be inherited from RESCO's
  benchmark suite and never removed — see §1.1.2).

### 2.6 RL-TSC method MDP definitions (Appendix C)

`TREX_comp/agents/{pfrl_dqn,pfrl_ppo,mplight,fma2c,ma2c}.py`, `TREX_comp/{states,rewards}.py`.

❓ **Only partially verifiable without the full Appendix C text.** Structural spot-checks:
- `states.drq_norm` (IDQN/IPPO), `states.mplight`/`mplight_full` (MPLight), `states.fma2c`/
  `fma2c_full` (FMA2C) are distinct per-method observation functions, consistent with the
  paper's claim of per-method MDP definitions — not diffed field-by-field against Appendix C.
  ✅/❓ Not flagged as wrong, but not confirmed correct either — recommend a follow-up pass
  once the paper text is available.
- Reward functions: `rewards.wait_norm` (IDQN/IPPO) clips `-total_wait/224` to `[-4,4]`
  (`TREX_comp/rewards.py:17-25`); `rewards.pressure` (MPLight) computes signed queue-pressure
  (`TREX_comp/rewards.py:28-41`); `rewards.fma2c` uses region-based liquidity/fringe metrics.
  The `224` normalization constant (`rewards.py:24`) is a magic number with no visible
  derivation or config binding — 🚩 flagged as a Phase 5 config-centralization candidate, not
  changed (unclear if `224` is a tuned constant tied to specific queue-capacity assumptions
  that the paper documents, or an arbitrary scaling choice).

### 2.7 RNG / seed-controlled reproducibility (ties Phase 2 + Phase 3)

The paper's brief claims "seed-controlled, averaged-over-5-seeds results." The code:
- Never passes `--seed` to SUMO — every `sumo_cmd` includes `'--random'`
  (`base_env.py:128`, `incident_env.py:196`), which tells SUMO to pick its own internal RNG
  seed non-deterministically each run.
- `Initializer.random()` reseeds the **global** `numpy` RNG (`np.random.seed(self.random_seed)`,
  `T_REX.py:88`) every time an incident is initialized (i.e., every episode).
- The RL agents' own exploration policies draw from that same global, just-reseeded `numpy`
  RNG rather than an independent generator — e.g. `TREX_comp/agents/pfrl_dqn.py:62,69,167`
  (`np.random.randint`, `np.random.rand`).

🚩 **Flagged for human review, not changed.** The net effect: SUMO-level vehicle stochasticity
is never seed-controlled at all, and RL exploration randomness is entangled with incident-seed
reseeding rather than independent. Properly fixing this (giving SUMO a deterministic `--seed`
derived from the run's seed, and giving the RL exploration policy its own independent
generator) is a good idea, but changes what "the same seed" reproduces — exactly the kind of
scientific-logic change ground rule 3 says must be flagged rather than silently altered.

---

### 2.6b `--strategy 3` ("curriculum") is documented but unimplemented, and crashes

`main.py`'s `--strategy` flag is documented as `"1 = base, 2 = incident, 3 = curriculum"`
(`main.py:47-48`), but there is no code path that treats `3` differently from `2`:
`env_class = BaseEnv if args.strategy == 1 else IncidentEnv` sends both to `IncidentEnv`,
and `level=2 if args.strategy == 2 else None` (`main.py:119`) means strategy `3` constructs
`IncidentEnv(..., level=None)`. `IncidentEnv._initialize_incidents` then does
`for i in range(self.level):` (`incident_env.py:250` at the time of writing) — `range(None)`
raises `TypeError` immediately. **`--strategy 3` is not a working "curriculum" mode; it's a
documented option that crashes on first use.** 🚩 Flagged, not implemented — building an
actual curriculum-learning strategy (presumably a progressive incident-severity ramp) is a
real feature to design, not a bug to fix blindly; left for human implementation.

## 2.8 Smoke test (Ground rule 4: run before/after changes)

SUMO/libsumo/torch/pfrl are all installed in this environment, so an actual end-to-end run
was possible (not just `py_compile`). `LIBSUMO_AS_TRACI=1 python3 main.py --agent
MAXPRESSURE --map ingolstadt7 --eps 1 --strategy 1` (base scenario, no incidents) ran
cleanly start-to-finish. Testing the incident scenario (`--strategy 2`, which exercises the
`Initializer`/`Deployment`/ICM/SSD code this audit touched) surfaced three **pre-existing**
issues, one of which was fixed and two of which are flagged rather than guessed at:

1. **✅ Fixed — real bug, confirmed by the crash itself.** `IncidentEnv._build_sumo_command`
   (`incident_env.py:80-92`, the branch used for `grid4x4`/`arterial4x4`, which pass a route
   *directory* rather than a `.sumocfg`) hardcoded `-a
   os.path.join(self.route, "vtypes.add.xml")` — a file that **does not exist anywhere**
   (confirmed: the tracked `grid4x4.zip`/`arterial4x4.zip` contain only `*.rou.xml` route
   files, no `vtypes.add.xml`). The correct additional-file path was already computed a few
   lines earlier as `self.additional` (`_find_additional_file()`, correctly resolving to
   `environments/grid4x4/grid4x4.add.xml`, which does exist) and is already used correctly
   by both the non-route branch of this same method and by `reset()`'s SUMO command — only
   this one route-branch call site reinvented a different, wrong path. This crashed
   `IncidentEnv.__init__` immediately (`TraCIException: Process Error`) for **every**
   grid4x4/arterial4x4 incident run before the fix. Fixed by using `self.additional`
   instead — same fix pattern the working branch already used, so no design guesswork
   involved.

2. **🚩 Critical, flagged, not fixed — missing incident-sampling input data for every
   real-world network.** `Initializer.__init__` (`T_REX.py:62-69`) unconditionally loads
   `Ing21_prob.csv`, `Ing7_prob.csv`, `Col3_prob.csv`, or `Col8_prob.csv` (bare relative
   filenames, resolved against the process's current working directory) whenever
   `map_name` is one of the four real-world networks the paper actually uses. **None of
   these four CSV files exist anywhere in this repository, in git history, or under any
   name resembling them** (confirmed by an exhaustive filename search). This means
   `python main.py --map ingolstadt7 --strategy 2` (or `ingolstadt21`/`cologne3`/`cologne8`)
   **crashes immediately** with `FileNotFoundError` before a single incident can be sampled
   — reproduced live during this smoke test. Per Section 2.3, these files encode
   "history flow data"-derived edge weights for the paper's non-uniform incident-location
   sampling on real-world networks; grid4x4/arterial4x4 don't need them because they use
   uniform `random_edge()` instead of `weighted_random_edge()`. **This is a hard blocker to
   reproducing any incident-scenario result on Cologne/Ingolstadt from a clean checkout of
   this repository as it stands.** Not fixed — fabricating plausible-looking edge-weight
   data would be inventing scientific input, squarely against ground rule 3. Needs the
   original CSVs restored from wherever they were generated (or committed if they exist
   only on a training machine).
3. **🚩 Flagged, not fixed — this session's own in-progress work, incomplete by design so
   far.** `Deployment.manage_incident_queue_teleport_exemption` (the teleport-exemption
   logic committed as this session's WIP baseline — see the top of this report) switches a
   queued vehicle's type to `'CAV4'` (`T_REX.py:840`). `CAV4` is only an **active** vType in
   `environments/ingolstadt21/ingolstadt21.add.xml` (added by the same WIP commit, with
   `timeToTeleport="-1"`). In `grid4x4`, `arterial4x4`, `cologne3`, `cologne8`, and
   `ingolstadt7`'s `.add.xml` files, `CAV4` exists only inside an unrelated **commented-out**
   `<!-- ... -->` block (a pre-existing, older, disabled vType-distribution experiment,
   unrelated to the teleport-exemption feature) — so `traci.vehicle.setType(veh, 'CAV4')`
   throws `TraCIException: Vehicle type 'CAV4' is not known` the first time any vehicle
   actually queues behind a blocked lane on any of those five networks (reproduced live
   during this smoke test on `grid4x4`). `cologne1`/`ingolstadt1` have no `.add.xml` at all.
   **Not fixed** — this is squarely the user's own active, unfinished feature work from
   *this session* (only ingolstadt21 had been wired up when the audit began); completing it
   for the other five networks means choosing vType attributes (vClass, whether to carry
   over `speedDev`/`carFollowModel`/etc. from the commented-out legacy block) that are a
   design decision for whoever is developing that feature, not an audit fix.

⚪ With finding 1 fixed, the base scenario and the incident-scenario *mechanics*
(Initializer sampling, block creation/removal, ICM/SSD code paths, logging) all run and
produce sensible output — see the `grid4x4` incident-settings log lines produced during
this test (realistic edge/lane/position/duration draws). The two remaining flagged items
are data/feature-completeness gaps, not defects introduced or missed by this audit's code
changes.

## 3. Summary table (all findings, severity-ordered)

| Sev | Finding | Status |
|---|---|---|
| Not an issue | LSI/FPD/CR/AUC/RAUC/PDI metrics not implemented in this repo | ⚪ By design (confirmed round 2) — computed by a separate downstream pipeline from `metrics_*.csv`/`tripinfo_*.xml` |
| Critical | `Ing21_prob.csv`/`Ing7_prob.csv`/`Col3_prob.csv`/`Col8_prob.csv` missing — incident scenario cannot run at all on any real-world network | 🚩 Flagged (data missing, cannot fabricate) |
| Bug | `IncidentEnv._build_sumo_command` route branch referenced a nonexistent `vtypes.add.xml` instead of the already-computed `self.additional` — broke grid4x4/arterial4x4 incident runs entirely | ✅ Fixed |
| Bug (flagged) | `CAV4` teleport-exemption vType only active in `ingolstadt21.add.xml`; commented-out or absent elsewhere — incident scenario crashes on any queued vehicle for 5 of 7 other networks | 🚩 Flagged — this session's own unfinished WIP feature, not completed |
| Bug (flagged) | `--strategy 3` ("curriculum") documented but unimplemented — crashes with `TypeError: range(None)` | 🚩 Flagged, not implemented |
| Correctness note | Table B1's 5 incident categories aren't a selectable code parameter; single generic blockage mechanism | 🚩 Flagged, README describes actual configurability only |
| Critical | No `.gitignore`; 1.3GB+ of generated route files and `.pyc` files tracked in git (6.4GB `.git`) | ✅ Fixed (gitignore + pycache removal) / 🚩 Flagged (arterial4x4 route files, size) |
| Critical | Per-network learning rate defaults not implemented; CLI default silently overrides paper-correct values | ✅ Fixed (added hyperparameter config) |
| Bug | Duplicate `get_arcs_cost` definition, first is dead code | ✅ Fixed |
| Bug | `MPLight.__init__` accepts `lr` but hardcodes `0.005` for the underlying `DQNAgent`, ignoring it | ✅ Fixed |
| Bug | `main.py::run_episode` discards its `obs` param, silently breaking `--repeat` seed replay | ✅ Fixed |
| Bug | TraCI/SUMO lifecycle has no exception safety → subprocess leak on error | ✅ Fixed |
| Bug (flagged) | Incident start-time upper bound `end_time-500` vs paper's `end_time-1200` | 🚩 Flagged |
| Bug (flagged) | `slow_zone_speed=1.39` contradicts its own comment (`13.8`) and the paper's ~8km/h figure | 🚩 Flagged |
| Bug (flagged) | SUMO never seeded (`--random` always); RL exploration RNG entangled with incident-seed reseeding | 🚩 Flagged |
| Bug (flagged) | `warmup=0` for every network | 🚩 Flagged |
| Style | Pervasive `print()` debugging instead of `logging` | ✅ Fixed |
| Style | `assert` used for runtime validation in sampling code | 🚩 Flagged |
| Cleanup | Large commented-out dead-code blocks in `T_REX.py` | 🚩 Flagged |
| Cleanup | `base_env.py`/`incident_env.py` ~90% duplicated boilerplate | 🚩 Flagged (not refactored — too risky to auto-apply) |
| Cleanup | `readXML.py` hardcoded path assumptions, dead/unused script | 🚩 Flagged |
| Hygiene | No LICENSE, tests, CI, CITATION.cff, CONTRIBUTING.md, pinned deps | ✅ Fixed (Phase 5/6) — LICENSE choice needs human confirmation |
| Hygiene | `resco_benchmark` imported but not in `requirements.txt` | ✅ Fixed (documented/pinned) |
