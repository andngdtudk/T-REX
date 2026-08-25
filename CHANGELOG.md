# Changelog

Changes made on the `audit/code-quality-and-docs` branch, grouped by audit phase. See
`AUDIT_REPORT.md` for the full analysis behind each item, including everything that was
**flagged but not changed** (most items below link back to a report section for that
reason). Nothing in the scientific formulas — metric definitions, ICM/SSD equations,
incident sampling distributions, RL reward/observation functions — was altered.

## Environment unification (`feature/unified-environment`, branched off `audit/code-quality-and-docs`)

Merged `base_env.py::BaseEnv` and `incident_env.py::IncidentEnv` — two separate,
drifting implementations of largely the same Gym environment — into one class,
`trex_env.py::TrexEnv`, with incidents toggled by a single `incident_config` parameter
(`None` = off, `IncidentConfig(...)` = on; `TREX_comp/config/incident_config.py`).
`main.py` now constructs `TrexEnv` unconditionally, selecting `incident_config` from the
existing `--strategy` flag (1→off, 2→on) — no new CLI flag added. `base_env.py`/
`incident_env.py` remain as thin, `DeprecationWarning`-emitting subclasses of `TrexEnv` for
anyone importing them directly; both verified live to still work end-to-end.

**Correctness verified, not assumed**: when incidents are disabled, `Initializer`/
`Deployment` are never constructed — confirmed by code inspection (every call site is
gated behind `if self.enable_incidents:`) and by the mandatory regression test below
showing zero divergence in output versus the pre-merge `BaseEnv`, which never touched the
incident subsystem either.

**Mandatory regression test** (`tests/test_env_unification.py`, requires SUMO, 3/3 passing
— see `AUDIT_REPORT.md` for the full pytest output and per-item discussion): runs the
frozen pre-merge `BaseEnv`/`IncidentEnv` (`tests/reference_impl/`) against the new
`TrexEnv` on `ingolstadt7`, same seed, same short episode, and diffs `metrics_1.csv` +
`tripinfo_1.xml`. Incidents-on: byte-identical, no normalization. Incidents-off: identical
after normalizing one known, documented, intentional cosmetic difference (see below) — not
a source of silent divergence.

**Bug found and fixed while diffing the two originals** (not preserved): pre-merge
`BaseEnv`'s route-file path construction (`self.route + '_N.rou.xml'`) disagreed with
pre-merge `IncidentEnv`'s and `main.py`'s own (`os.path.join(self.route, ...)`) — confirmed
live that pre-merge `BaseEnv` could not run `--strategy 1` on `grid4x4`/`arterial4x4` at all
given the documented decompression layout (`TraCIException: route file ... not accessible`).
This went undiscovered until now because no prior smoke test exercised `--strategy 1` on a
route-based network. `TrexEnv` uses the working (subdirectory) convention for both modes.

**Other edge cases found and intentionally handled** (full detail in `AUDIT_REPORT.md`):
`save_metrics`'s CSV formatting differed by one trailing-comma byte between the two
originals (unified onto the cleaner format); additional-file loading genuinely differs by
design between incidents-on/off and is kept that way, not converged; phase-string filtering
and results-directory path construction differed cosmetically between the originals with no
observed effect (confirmed by the regression test), and were unified for cleanliness.

**Follow-up**: `--time-to-teleport -1` (initially kept conditional on `incident_config`,
matching each pre-merge original exactly) has been removed from both modes per repo-owner
direction — `CAV4`'s per-vehicle teleport exemption already covers what the blanket global
override was for, so it served no purpose even in the "incidents off" path where it was only
ever inherited legacy `BaseEnv` behavior. Re-verified live: the regression test suite still
passes 3/3 after this change.

## Round 2 — follow-up to round 1's flagged items

Round 1 flagged several items rather than guessing; round 2 resolves the ones that had a
clear, safe answer and re-verifies the rest live rather than assuming they're fixed.

- **Incident-probability CSVs restored** (`Ing21_prob.csv`, `Ing7_prob.csv`, `Col3_prob.csv`,
  `Col8_prob.csv`) — moved into their respective `environments/<network>/` directories
  (they arrived at repo root) and `Initializer.__init__` now resolves them relative to the
  network's own directory instead of the process's CWD (a bare filename would only have
  worked if `main.py` happened to be launched from the exact directory containing them).
  Re-verified live: all four real-world networks (cologne3, cologne8, ingolstadt7,
  ingolstadt21) get past the incident-sampling step now.
- **Metrics (LSI/FPD/CR/AUC/RAUC/PDI) reclassified from "critical gap" to "by design, out
  of scope"** per repo-owner clarification: they're computed by a separate downstream
  analysis pipeline, not part of T-REX. `AUDIT_REPORT.md` and a new README "Metrics &
  analysis" section now document exactly what T-REX itself logs per episode
  (`metrics_<run>.csv`, `tripinfo_<run>.xml`) instead.
- **CAV4 teleport-exemption vType generalized to all six incident-capable networks**
  (previously only `ingolstadt21`) — uncommented/added the identical
  `<vType id="CAV4" .../>` definition already working there to `grid4x4`, `arterial4x4`,
  `cologne3`, `cologne8`, `ingolstadt7`. Added `tests/test_incident_vtypes.py`, which parses
  every incident-capable network's `.add.xml` (no SUMO needed) and fails if any of them is
  missing an active CAV4 vType, so this can't silently regress to one-network-only again.
  Re-verified live on all four real-world networks: none crash on `TraCIException:
  Vehicle type 'CAV4' is not known` anymore.
- **`--strategy 3` ("curriculum")** now raises a clear `NotImplementedError` instead of
  crashing opaquely inside `IncidentEnv` with `TypeError: 'NoneType' object cannot be
  interpreted as an integer`; `argparse` and the README mark it "planned, not yet
  implemented" rather than listing it as supported.
- **End-to-end SUMO seed control added**: `--seed` (default `42`) now seeds Python's
  `random`, `numpy`, and `torch`, and is threaded through to `BaseEnv`/`IncidentEnv`, which
  launch SUMO with `--seed <seed + episode_number>` instead of always `--random`.
  `--no-seed-sumo` restores the old `--random` SUMO behavior while still seeding
  Python/numpy/torch. Verified live: two independent runs with `--seed 7` produce
  bit-identical incident sampling. The deeper RNG-architecture question (the RL agents'
  exploration policies sharing the same global numpy RNG that `Initializer` reseeds every
  episode) is a separate, larger change and was **not** touched here.
- **Added a "Needs owner decision" table** in `AUDIT_REPORT.md` for the three items that
  genuinely can't be resolved without the manuscript: the incident start-time sampling
  upper bound (`end_time−500` vs. the paper's `end_time−1200`), `warmup=0` in every network
  config, and the `slow_zone_speed` value contradicting its own inline comment. None of
  these were changed.
- 31/31 tests pass (up from 19: +6 CAV4-vtype regression tests, one per incident-capable
  network, +6 SUMO seed-arg tests). All five smoke-tested networks (grid4x4 + the four
  real-world networks) run `--strategy 1` and `--strategy 2` end-to-end without a traceback.

## Phase 1 — Repository hygiene

- Added `.gitignore` (Python bytecode, SUMO run artifacts, editor/OS files, decompressed
  route directories); removed 47 already-committed `__pycache__`/`.pyc` files from version
  control (`TREX_comp/`).
- Started `AUDIT_REPORT.md` with the full static-analysis and dependency-management
  findings (unpinned `requirements.txt`, undocumented SUMO/RESCO versions, 1.3GB+ of
  generated route files and zip archives in `environments/arterial4x4`/`grid4x4`, no
  tests/CI/LICENSE/CITATION prior to this audit).

## Phase 2 — Correctness audit against the paper

- Diffed the ICM rerouting model, SSD speed-adaptation formula, incident sampling
  distributions, and available hyperparameters against the paper's stated equations/values
  (as summarized in the audit task brief — the full manuscript text wasn't available to
  this audit; see `AUDIT_REPORT.md`'s caveat at the top).
- Confirmed exact matches: ICM `β0/β_gain/β_loss`, AASHTO SSD constants (`t=2.5s`,
  `a=3.4 m/s²`), incident position `U(10, edge_length−10)`, incident duration
  `~Exp(0.029)`, driver-heterogeneity mix and per-type error rates, `γ=0.99` for IDQN/MPLight.
- Flagged (not changed, per the ground rules — these need a human with the manuscript to
  adjudicate): incident start-time upper bound (`end_time−500` in code vs. `end_time−1200`
  in the brief), `warmup=0` for every network, `slow_zone_speed` contradicting its own
  code comment, SUMO never being seeded (`--random` always) alongside RL exploration RNG
  sharing state with incident-seed reseeding, and the fact that the paper's five incident
  *categories* (Table B1) aren't a selectable parameter in the current single generic
  blockage mechanism.
- **Discovered the LSI/FPD/CR/AUC/RAUC/PDI robustness metrics are not implemented anywhere
  in this repository** — flagged as the most significant scope gap found by this audit.

## Phase 3 — Bug fixes

- Removed a dead, unreachable duplicate definition of `Deployment.get_arcs_cost` in
  `T_REX.py` (Python was silently using only the second of two same-named methods).
- Fixed `MPLight.__init__` silently ignoring its `lr` parameter (hardcoded `lr=0.005` when
  constructing the underlying `DQNAgent` instead of passing through the value it received).
- Fixed `main.py::run_episode` discarding its own `obs` parameter, which silently broke the
  `--repeat` fixed-incident-seed testing/replay feature (every call re-reset the environment
  with a fresh random seed instead of honoring the one the caller had already set up).
- Fixed `IncidentEnv._build_sumo_command`'s route-based branch (grid4x4/arterial4x4)
  referencing a nonexistent `vtypes.add.xml` instead of the already-correctly-computed
  `self.additional` — this crashed `IncidentEnv.__init__` immediately on those two networks;
  found and confirmed via an end-to-end smoke test.
- Made SUMO/TraCI start-teardown sequences exception-safe (`try/finally`) in `base_env.py`,
  `incident_env.py`, and `main.py::run_trial`, so a mid-episode exception no longer leaks
  the SUMO subprocess.
- Fixed two debug messages in `T_REX.py` that were missing their `f`-string prefix and so
  never actually interpolated the variable they were meant to show.
- Replaced `print()` debugging with `logging` (DEBUG/INFO/WARNING/ERROR as appropriate) in
  `T_REX.py`, `base_env.py`, `incident_env.py`, and `main.py`; added a `--verbose` CLI flag.
  Logged (instead of silently swallowing) the broad `except Exception` in `readXML.py`'s
  tripinfo parser.
- Removed unused imports (`os`, `sys`, `json`, `pandas`, `xml.etree.ElementTree`,
  `optparse`, `sumolib.checkBinary`, `time.time` from `T_REX.py`; `pathlib.Path` from
  `main.py`).

Found via the smoke test but **not fixed** (flagged for human follow-up, see
`AUDIT_REPORT.md` Section 2.8): the incident-location-probability CSVs
(`Ing21_prob.csv`/`Ing7_prob.csv`/`Col3_prob.csv`/`Col8_prob.csv`) that weighted incident
sampling requires are missing from the repository entirely, blocking any incident-scenario
run on a real-world network from a clean checkout; and the `CAV4` teleport-exemption vType
(this session's own in-progress WIP feature) is only wired up in `ingolstadt21.add.xml` so
far. Also found: `--strategy 3` ("curriculum") is documented in `main.py`'s CLI help but has
no implementation and crashes immediately (`AUDIT_REPORT.md` Section 2.6b).

## Phase 4 — Cleanup

- Cataloged (but did not remove, pending human sign-off) large commented-out dead-code
  blocks in `T_REX.py`, and the code duplication between `base_env.py`/`incident_env.py`
  and across `TREX_comp/agents/*.py` — see `AUDIT_REPORT.md` Sections 1.3/1.5 for exact
  locations and why an automatic refactor wasn't applied to code this close to the paper's
  published results.

## Phase 5 — Software engineering standardization

- Extracted `Deployment.calculate_ssd(speed)` as a pure, unit-testable classmethod (same
  arithmetic `speed_adjustment` already computed inline; also turns the inline AASHTO
  constants into named class attributes).
- Added `tests/` (pytest): SSD formula, ICM binomial-logit rerouting probability and its
  expected-gain/avoided-loss components, and `Initializer`'s random incident-sampling
  bounds/statistics — 19 tests, none requiring a live SUMO connection (see `tests/README.md`
  for how `Initializer`/`Deployment` are tested without their normal SUMO-dependent
  constructors).
- Added `TREX_comp/config/hyperparams.py` with the paper's per-network IDQN/MPLight
  learning-rate defaults (Appendix B) and wired it into `main.py`, replacing the previous
  behavior where every run silently used a single CLI-wide default (`0.001`) regardless of
  network.
- Pinned `requirements.txt` to minimum-compatible versions (previously fully unpinned);
  added `requirements-dev.txt` and `pyproject.toml` (ruff/black/isort/pytest config).
- Added `.github/workflows/ci.yml`: pytest as a required check; ruff/black/isort as
  advisory-only for now (the pre-existing codebase hasn't had a full lint/format pass
  applied — see `AUDIT_REPORT.md` Section 1.4 for why that wasn't bundled into this audit).
- Added `LICENSE` (MIT, user-selected during this audit), `CITATION.cff`,
  `CONTRIBUTING.md`.
- **Not done** (see `AUDIT_REPORT.md` Section 1.9 for the reasoning): moving to a `src/`-style
  package layout, and a full `T_REX.py` module split — both would touch every import in the
  training pipeline and were judged too risky to do unreviewed in this pass.

## Phase 6 — README rewrite

- Rewrote `README.md`: badges, an accurate description tied to what's actually in this
  repo (not the four-module architecture in the abstract alone), an architecture diagram,
  installation/quickstart instructions validated against a real smoke test, an honest
  "reproducing the paper's experiments" mapping (`main.py` flags, not separate per-experiment
  scripts that don't exist), a corrected "configuring incidents" section (no selectable
  incident-category parameter exists in code), supported-networks table, and citation
  (using the real author names/arXiv ID already present in the pre-audit README — no
  double-blind placeholder text was found to remove).
