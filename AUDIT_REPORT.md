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

## Round 5 — Part A: closing the round-1 trust gap (live re-verification)

Round 1 marked several items "✅ Fixed" based on the code change being made, not on a live
re-run confirming the fix actually works at runtime — the same class of claim that turned
out stale for the CSV finding (§4.2). Each item below was re-run live this round; results
and evidence follow, not restated claims. All six: **PASS**.

### A.1 `print()` → `logging` conversion — PASS

`ingolstadt7`, `--strategy 2`, one episode, with vs. without `--verbose`:
```
no-verbose DEBUG count: 0    no-verbose INFO count: 15
verbose    DEBUG count: 282  verbose    INFO count: 15
```
`--verbose` genuinely gates the logging level at runtime (0 → 282 DEBUG lines appearing,
identical INFO-level output either way) — not merely that `print()` calls were textually
replaced.

### A.2 TraCI/SUMO exception safety (`try/finally`) — PASS

Live check (not part of the permanent suite, see below for that): constructed `TrexEnv`
with `libsumo=False` (so a real `sumo` OS subprocess is spawned, making an orphan-process
check meaningful) and a `reward_fn` that raises on its 2nd call, wrapped in the same
`try/.../finally: env.close()` pattern `main.py` uses:
```
PIDs before: []
PIDs after construction: []
PIDs after reset: ['397827 .../sumo/bin/sumo -c .../ingolstadt7.sumocfg ... --remote-port 36975']
Exception raised as expected: deliberate mid-episode failure for exception-safety test
PIDs after close(): []
RESULT: PASS -- no orphan process
```
A real subprocess is confirmed running mid-episode, then confirmed **gone** after `close()`
runs from the `finally` block, despite the exception. Permanent regression test added:
`tests/test_exception_safety.py` (uses `libsumo=True` for CI speed; checks that `close()`
itself doesn't raise a second error while unwinding, and that a second, independent env can
be constructed afterward — proving libsumo's single global simulation slot was actually
released, the fast-mode proxy for "no orphan state left behind").

### A.3 `MPLight` `lr` bug fix — PASS

Constructed real `MPLight` agents (PyTorch, no SUMO needed) at several learning rates and
read the actual optimizer state, not the source line:
```
requested lr=0.005  ->  optimizer lr=0.005  ->  MATCH
requested lr=0.001  ->  optimizer lr=0.001  ->  MATCH
requested lr=0.05   ->  optimizer lr=0.05   ->  MATCH
```
Permanent test: `tests/test_mplight_lr.py` (5 tests, parametrized over 4 additional lr
values including a very small `1e-5`).

### A.4 `run_episode` `obs` fix / `--repeat` seed replay — PASS

The real test of this fix, run via the actual CLI end to end: trained `MAXPRESSURE` on
`ingolstadt7` for 3 episodes with `--repeat 1 --seed 42`, then tested with `--load True
--repeat 1` using a **different** top-level `--seed 999` (deliberately, to prove replay is
driven by the saved incident seed file, not by the outer seed coincidentally matching).

Training, last (saved) episode:
```
INFO T_REX: Incident happens at edge 168702040#2 at time 2530 lasting for 3840 seconds, lanes=[0, 1], pos=46.90258195942547, random_seed=1924204410
INFO T_REX: Incident happens at edge 402600768#1 at time 860 lasting for 480 seconds, lanes=[0, 1, 2], pos=16.12940031325934, random_seed=1159652549
INFO __main__: Saved seeds to MAXPRESSUREingolstadt7-seed_ic1.txt: [1924204410]
INFO __main__: Saved seeds to MAXPRESSUREingolstadt7-seed_ic2.txt: [1159652549]
```
Testing/replay (`--seed 999`, i.e. a different top-level seed):
```
INFO __main__: Loaded seeds from files: [1924204410], [1159652549]
INFO T_REX: Incident happens at edge 168702040#2 at time 2530 lasting for 3840 seconds, lanes=[0, 1], pos=46.90258195942547, random_seed=1924204410
INFO T_REX: Incident happens at edge 402600768#1 at time 860 lasting for 480 seconds, lanes=[0, 1, 2], pos=16.12940031325934, random_seed=1159652549
```
Edge, time, duration, lanes, and position are byte-identical between the recorded and
replayed episode. Permanent regression test: `tests/test_repeat_seed_replay.py` (checks
`env.run` doesn't increment a second time when `run_episode` is given a pre-seeded `obs` —
a second, hidden `reset()` is exactly the failure mode that broke this).

### A.5 `get_arcs_cost` duplicate removal — PASS

Only one definition remains (`T_REX.py:1337`, confirmed by grep). The existing
`tests/test_icm.py` suite already indirectly validated the surviving (downstream) semantics
via `calculate_avoided_loss`/`calculate_expected_gain`, but didn't call `get_arcs_cost`
itself — added direct coverage: `test_get_arcs_cost_returns_downstream_travel_times` (mocks
a fake network, confirms it returns the correct downstream travel times, non-NaN,
non-negative) and `test_get_arcs_cost_empty_when_no_downstream_edges`. 9/9 passing in
`tests/test_icm.py` now (was 7).

### A.6 Dependency/requirements claims — PASS, with one caveat worth noting

Created a genuinely clean virtualenv (`python -m venv`, no packages carried over), installed
`pip install -r requirements.txt` (83s, no errors), and ran the README's exact documented
verify-install command (`export LIBSUMO_AS_TRACI=1; python main.py --agent MAXPRESSURE --map
grid4x4 --eps 1 --strategy 1 --libsumo True`) against that clean interpreter — completed
successfully, metrics written.

**Caveat**: `libsumo` (the fast in-process SUMO binding, as opposed to `eclipse-sumo`'s
subprocess-based `traci`) is not itself listed in `requirements.txt` and was not installed
by it. In the clean venv, `LIBSUMO_AS_TRACI=1` triggered a `UserWarning: Could not import
libsumo ... falling back to pure python traci` rather than an error — the run still
succeeded (just slower, via a real spawned subprocess instead of in-process). 🚩 Not a
failure, but worth a README/requirements note: on a machine without `libsumo` separately
installed, `--libsumo True` silently degrades to subprocess-based TraCI rather than actually
using libsumo, contrary to what the flag name and `LIBSUMO_AS_TRACI` env var suggest. Not
fixed this round (out of Part A's re-verification scope, and not one of the six items asked
for) — noted for a future pass.

## Round 5 — Part B: manuscript-confirmed fixes

The repo owner directly read the submitted manuscript and confirmed the following against
the actual paper text (not the audit brief's paraphrase). No longer ambiguous "code vs.
brief" disputes — applied as confirmed fixes, each citing its exact paper section in both
the code comment and the commit message.

### B1. Incident start-time upper bound → `end_time − 1200` — ✅ Applied

Manuscript Section 2.3: `t_start ~ U(t_warmup, t_end − 1200)`. `T_REX.py::random_time()`
changed from `end_time - 500`. Live-reverified on `ingolstadt7`: every sampled start time
now falls within the tightened window (`≤ 2400` for a 3600s episode). Test updated:
`tests/test_incident_sampling.py::test_random_time_within_paper_bounds`.

### B2. Warm-up phase → `100` seconds — ✅ Applied

Manuscript Section 3.5: "Each traffic episode simulates 3,600 seconds, including a
100-second warm-up phase." `TREX_comp/config/map_config.py`: `warmup` changed from `0` to
`100` in all 10 network entries. Confirmed uniform, not network-specific: every entry's
`end_time − start_time` is exactly 3600s, matching the manuscript's blanket per-episode
duration — no exception to carve out.

### B3. `slow_zone_speed` → `2.2352` m/s (exact 5 mph) — ✅ Applied

Manuscript Section 2.4.2: "a conservative reduced speed of 5 mph (approximately 8 km/h)."
`T_REX.py:28`: `self.slow_zone_speed` changed from `1.39` to `2.2352` (`5 * 1609.344 / 3600`,
the exact conversion — chosen over a rounded value so the constant is traceably derived, not
an approximation of an approximation). This replaces the uncommitted WIP edit found at the
very start of this audit, whose own comment (`# 13.8 is 50 km/h`) didn't match the value it
was attached to (`1.39`) either — both the value and the stale comment are now corrected
together. New test: `tests/test_ssd.py::test_slow_zone_speed_matches_paper_5mph` (reads
`Initializer.__init__`'s source directly via `inspect.getsource`, since the class needs a
live SUMO connection to construct fully and this value isn't extracted into a testable class
attribute).

### B4. SUMO seed control + exploration/incident RNG independence — ✅ Applied (deliberate methodology change, not a bug fix)

Manuscript Section 3.4 confirms results are "averaged over five random seeds" — full-pipeline
seed control is part of the paper's stated methodology, not optional. Two parts:

1. **SUMO-level `--seed`** (already implemented in earlier rounds — `main.py`'s `--seed`
   flag, default `42`, threaded through `TrexEnv._seed_args()` to SUMO's `--seed` instead of
   `--random`; see rounds 1-2). No new code needed here, already live and tested.
2. **Independent exploration RNG** (new this round): `main.py::run_trial` now constructs
   `agt_config['exploration_rng'] = np.random.default_rng(args.seed)` before agent
   construction — a `Generator` instance using a mathematically distinct algorithm (PCG64)
   from the legacy global `numpy.random` API (Mersenne Twister) that
   `T_REX.py::Initializer.random()` reseeds every episode via `np.random.seed(...)`, so
   sharing the same seed *value* creates no correlation between the two streams.
   - `TREX_comp/agents/pfrl_dqn.py::DQNAgent` (used by IDQN and MPLight): reads
     `config.get('exploration_rng')` (falls back to a fresh, unseeded `Generator` if absent,
     e.g. direct construction in tests), uses it for the random-action-selection lambda in
     both the shared (`SharedEpsGreedy`, MPLight's path) and non-shared
     (`explorers.LinearDecayEpsilonGreedy`, IDQN's path) explorers, and for the
     epsilon-vs-random coin flip in `select_action_epsilon_greedily`
     (`SharedEpsGreedy`/MPLight's path only — see caveat below).
   - `TREX_comp/agents/ma2c.py::MA2CAgent` (FMA2C): same pattern, used for the
     `np.random.choice` policy-action sampling in `act()`.
   - **Not touched**: `ma2c.py`'s `ortho_init` (`np.random.standard_normal`) is one-time
     TensorFlow *weight initialization* at model-build time, not an exploration/action-
     selection call — a structurally different concern (fires once, not every training
     step) and out of scope for "exploration vs. incident sampling" entanglement.
   - **Caveat, honestly scoped, not overclaimed**: IDQN (non-shared path) only controls the
     *which-action-to-explore* draw via the injected `Generator` — the *whether-to-explore*
     epsilon coin flip happens inside `pfrl`'s own `LinearDecayEpsilonGreedy.select_action`
     (third-party library internals, not reachable without forking/monkeypatching pfrl,
     which is out of scope). MPLight (shared path, our own `SharedEpsGreedy` subclass) and
     FMA2C are **fully** decoupled — every exploration-relevant random draw goes through the
     injected `Generator`.

**Proof, not assumed** (per instruction, run and confirmed rather than left as a claim):
- Mathematical/isolated proof: same-seed `Generator`s produce identical draws; different
  seeds diverge; and — the critical check — interleaving global `np.random.seed()` reseeds
  (exactly what `Initializer.random()` does every episode) between draws from an independent
  `Generator` does **not** perturb that Generator's own sequence. All three confirmed both
  in an ad hoc script and as a permanent test using the *real* production code paths (not a
  simulation): `tests/test_rng_independence.py`, which interleaves actual
  `T_REX.py::Initializer` construction/reseeding with actual `DQNAgent` explorer draws.
  3/3 passing.
- Live full-pipeline re-verification: `IDQN` and `MPLight` (the shared-explorer path, the
  more heavily modified one) both trained 2 episodes end-to-end on `ingolstadt7` with the
  new RNG threading in place — no errors, metrics written, no lingering SUMO processes
  either way.

This **does** change simulation/training output relative to the prior unseeded-SUMO,
entangled-exploration behavior — expected and correct per the manuscript's stated
methodology, not a regression. Distinct from B1-B3 (which correct code that diverged from
the paper's own stated formulas/constants): B4 is a deliberate reproducibility capability
the paper's methodology requires and the code previously lacked, not a wrong-value bug.
Anyone diffing pre- and post-this-round output on a fixed seed will see a real difference in
both incident placement (already true since round 1-2's SUMO-seed work) and now also agent
exploration — that's the point, not a surprise to be alarmed by.

### B5. 20m short-edge threshold — confirmed consistent with the manuscript's own reasoning — no code change

Manuscript Section 2.3: "the 10-meter buffer at each end prevents bugs in SUMO." Round 4's
fix (excluding edges `<20m` — two 10m buffers — from the incident-candidate pool, see §4.4)
is directly consistent with this stated reasoning: the manuscript itself motivates the
10m-per-end buffer that makes edges shorter than 20m infeasible. No code change needed;
closing this "Needs owner decision" item as resolved by the manuscript text itself rather
than left open.

## Round 4 — verification, attribution, and one real bug

### 4.1 `arterial4x4` `rm -rf` recovery — verified clean, evidence below (not restated)

Actual command output, this session, repo root:

```
$ git status --porcelain environments/arterial4x4/
(empty)
$ git diff --stat HEAD -- environments/arterial4x4/
(empty)
$ git log --oneline -3 -- environments/arterial4x4/
ce3552646 fix: restore incident-probability CSVs and generalize CAV4 exemption
9d2c5285d Initial upload of full T-REX framework
```

Both status and diff are empty — genuinely clean, not merely "recovered." `git log` confirms
the file history touches exactly two commits: the original repo upload (route files
themselves) and round 2's unrelated `CAV4` addition to `arterial4x4.add.xml` — nothing from
the `rm -rf`/recovery cycle left a trace, because the recovery (`git checkout --
environments/arterial4x4/`) ran before anything was staged.

**What triggered it:** during round 3's 8-network CAV4 smoke test, both `grid4x4.zip` and
`arterial4x4.zip` were decompressed (`unzip -d <name>`) to exercise the two route-based
networks. Cleanup afterward ran:
```
rm -rf environments/grid4x4/grid4x4 environments/arterial4x4/arterial4x4
```
in a single combined command, on the (wrong) assumption that both decompressed directories
were equally disposable test artifacts. `grid4x4/grid4x4/` genuinely was (0 tracked files,
gitignored). `arterial4x4/arterial4x4/` was not — it's ~2,800 files committed as part of the
original repo upload (`9d2c5285d`, before this audit began; see §1.1.2). The mistake was
treating "I just decompressed this" as equivalent to "this is untracked," without checking.

**Guardrail for future rounds:** before running any recursive delete against a path inside a
git-tracked directory, run `git ls-files <path> | head` first — an empty result means it's
safe to `rm -rf`; any output means `git rm` (or leave it alone) instead. Never assume a
directory is disposable just because *this session* created it by decompressing something —
check whether it happens to coincide with an already-tracked path.

### 4.2 Round 2 vs. round 3 CSV discrepancy — reconciled

**One-sentence answer:** this is case (a) — round 2's audit report was generated from
observations made *before* the repo owner supplied the four CSV files and before round 2's
own fix commit (`ce3552646`) landed, and that pre-fix language was never revised afterward,
even though later text in the same round-2 report *does* correctly describe the fix (compare
§2.4/§2.8's original wording, written first, against the CSV item's "Update (round 2/3)"
callout added just above it) — so the contradiction is a stale-narration issue within round
2's own report, not a stale checkout, wrong branch, or genuinely different repo state; `git
log` shows a single continuous line of commits with no branch-switching in between.

**Caveat this implies:** if round 2's report narrated at least one fix (CSVs) as still-broken
after having already fixed it in the same round, its other "🚩 Flagged, not fixed" items from
that same round should not be trusted at face value without a similar live re-check — they
were not re-verified in this round (out of scope), but a future round should re-run the
underlying commands for round 2's remaining flagged items before relying on their stated
status, the same way round 3 did for the CSVs and CAV4.

### 4.3 The uncommitted `IC` vType edits — isolated, not assumed

Per round 3's note, checked which commit these landed in and isolated the exact diff:

```
$ git log --oneline --all -- environments/ingolstadt7/ingolstadt7.add.xml
d9385df27 fix: complete CAV4 teleport exemption for all 8 networks; fix stale gitignore
ce3552646 fix: restore incident-probability CSVs and generalize CAV4 exemption
9d2c5285d Initial upload of full T-REX framework
```

They are already committed — inside `d9385df27` (round 3's CAV4-completion commit), mixed in
with round 3's own `CAV4` additions to the same 5 files. **Not separated retroactively** (no
history rewrite, per instruction); the exact lines are documented here instead so the repo
owner can review them directly:

```diff
-	<vType id="IC" vClass="emergency" />
+	<vType id="IC" vClass="emergency" timeToTeleport="-1"/>
```
in `environments/{grid4x4,arterial4x4,cologne3,cologne8,ingolstadt7}/*.add.xml` (5 files);
also present by construction in the two new files this round created,
`environments/{cologne1,ingolstadt1}/*.add.xml`, since both were copied verbatim from
`ingolstadt21.add.xml`, which already had this line before this audit began (part of the
original WIP baseline committed at the very start, `5ad043ab6`). **Origin: not this audit.**
Round 3 found these 5 files' `IC` lines already modified, uncommitted, in the working tree
at the start of that round — nobody in this audit wrote that diff; it was folded into round
3's commit only because it was sitting in the same files being edited for `CAV4`, not because
it was verified or authored here.

> **Update (round 6):** confirmed intentional by the repo owner — part of the
> teleport-exemption feature (the same mechanism `CAV4` implements: neither the incident's
> blocking dummy vehicle itself, typed `IC`, nor a genuinely-queued regular vehicle, typed
> `CAV4`, should be silently removed by SUMO's global teleport timeout while an incident is
> active). No longer listed under "Needs owner decision" — the commit that originally
> flagged it as unconfirmed (`d9385df27`) is not rewritten (no history rewrites, per ground
> rules); this note supersedes that flag with the confirmed status instead.

### 4.4 Fixed: `Initializer.random_pos()` could return a negative position

Confirmed genuine bug (not a paper-vs-code ambiguity — a negative position is physically
invalid under any reading): `random_pos()` called `np.random.uniform(10, edge_length - 10)`
unconditionally; when `edge_length < 20`, the upper bound is below the lower bound, which
NumPy documents as undefined behavior for `uniform()`. Fixed per the specified default
policy: edges shorter than **20m** (the exact threshold — two 10m end buffers with zero room
between them) are now excluded from the incident-candidate pool in both `random_edge()` and
`weighted_random_edge()`, rather than clamping the sampled position into a degenerate range.
This preserves `U(10, edge_length-10)` as a true uniform draw on every edge that remains
eligible, instead of silently distorting it on short edges. See `tests/test_incident_sampling.py`
for the regression test (a deliberately 15m synthetic edge, asserting it's excluded from the
candidate pool, plus a bounds check across a range of edge lengths including the 20m boundary
itself). 🚩 The 20m threshold and exclusion-vs-clamping choice are flagged in "Needs owner
decision" below as an overridable default, not a final scientific-methodology call — shipped
now because a live invalid negative position is strictly worse than a conservative default.

**Live re-verification**, `--map ingolstadt7 --strategy 2 --eps 5 --seed 1` (the exact
network the bug first appeared on): 12 incidents sampled across 5 episodes, positions
`32.4, 31.7, 11.2, 42.6, 36.2, 13.1, 57.6, 39.3, 167.7, 22.7, 18.0, 47.1` — all positive,
none negative. `tests/test_incident_sampling.py`: 12/12 passing (4 new tests for this fix).

**Reproducibility side effect, worth knowing about:** this fix necessarily changes which
edges are eligible for incident placement, so for networks with any short edges, the same
`--seed` value no longer selects the same incident edge/position it did before this fix
(confirmed live: re-running `ingolstadt21` with the same seed used earlier in this audit
selected different edges post-fix than pre-fix, both deterministically reproducible on their
respective sides of the fix). This is an unavoidable consequence of correctly excluding
invalid edges rather than a new bug — any incident-scenario results already generated with a
fixed seed on a network with short edges will not reproduce byte-for-byte against this
version of the code. Not flagged as a decision item (the fix itself isn't in question, only
its exact threshold/strategy are, per item 7 below) — noted here so it doesn't surprise
anyone diffing old vs. new seeded runs.

## Needs owner decision — consolidated (all rounds, single authoritative list)

**Resolved in round 5** (manuscript-confirmed by the repo owner directly, see Part B
above) — no longer open:
- ~~Incident start-time upper bound~~ → fixed to `end_time − 1200` (B1, Section 2.3).
- ~~`warmup=0`~~ → fixed to `100` (B2, Section 3.5).
- ~~`slow_zone_speed=1.39`~~ → fixed to `2.2352` (B3, Section 2.4.2).
- ~~SUMO-level `--seed` vs. RL-exploration-RNG independence~~ → both implemented (B4,
  Section 3.4) — SUMO seeding in earlier rounds, exploration-RNG independence this round.
- ~~`random_pos()`'s 20m threshold, confirmed vs. speculative~~ → confirmed consistent with
  the manuscript's own stated reasoning (B5, Section 2.3) — the threshold itself was never
  in question, only whether it matched the paper's rationale; it does.

**Resolved in round 6** — the two repo-policy items settled by direct owner decision:
- ~~Whether to keep, regenerate, or remove `arterial4x4`/`arterial5x5`~~ → **decision:
  keep, tracked as-is.** Confirmed unchanged by round 4's `.gitignore` fix
  (`git ls-files environments/arterial4x4/ | wc -l` → `2806`, matching round 4's original
  finding exactly; `environments/arterial5x5/` → `0`, confirmed still empty/no data). Kept
  despite being confirmed unused by the paper's published experiments (manuscript Section
  3.1, added as evidence in round 5) — presumably for extensibility or other users'
  benefit; the repo owner's call to make, and it's been made. README's "Supported networks"
  section notes both networks are retained but weren't part of the published results.
- ~~The uncommitted `IC` vType `timeToTeleport="-1"` edits~~ → **confirmed intentional**,
  part of the teleport-exemption feature (round 6). The commit that originally flagged this
  as "origin unconfirmed" (`d9385df27`, round 3) is not rewritten — per ground rules, no
  history rewrites — this entry now records the superseding confirmed status instead.

**Still open**, waiting on the repo owner:

| # | Item | File(s) | Since |
|---|---|---|---|
| 1 | Round 2's other 🚩-flagged items (beyond the CSV finding, which round 3 re-verified, and B1-B5 above) were never re-verified against a live run — no specific bug identified, just an open trust gap worth a future pass, in the same spirit as round 5 Part A's re-verification of round 1's claims. | (any remaining round 2 flagged items not covered above) | round 4, unchanged rounds 5-6 |

## Round 3 — validation, then completion of skipped items

Round 3 opened by pointing out that round 2's write-up and the actual repo state appeared
to contradict each other on three items (CSVs, CAV4, `--strategy 3`). Investigated each
directly rather than trusting either side; results below, evidence included, not assumed.

### 3.1 The CSV "contradiction" — resolved: the files are present and working

**Static check** (`git branch --show-current` → `feature/unified-environment`;
`find / -iname "*prob*.csv"` and `find <repo> -iname "*.csv"`, both from a clean shell):
all four files exist, inside the repo, in the correct per-network directories:

```
environments/ingolstadt7/Ing7_prob.csv
environments/cologne8/Col8_prob.csv
environments/cologne3/Col3_prob.csv
environments/ingolstadt21/Ing21_prob.csv
```

`git ls-files | grep -i prob.csv` confirms all four are **tracked and committed**
(commit `ce3552646`, already on this branch) — not merely present on disk. `T_REX.py`'s
`Initializer.__init__` (current lines ~57-64) resolves them via
`network_dir = os.path.dirname(self.net_path)` — already fixed in that same commit to be
independent of the process's working directory, not the bare-CWD-relative filename the
round-3 brief described. (`git log --oneline -- T_REX.py` confirms `ce3552646` is the
commit that introduced this fix; nothing since has touched it.)

**Live re-run**, the actual command that round 2 reported failing, from a clean shell, repo
root, `feature/unified-environment` checked out:

```
$ export LIBSUMO_AS_TRACI=1
$ python3 main.py --agent MAXPRESSURE --map ingolstadt7 --eps 1 --strategy 2 --libsumo True
...
INFO T_REX: Incident happens at edge 164051413 at time 649 lasting for 0 seconds, ...
INFO T_REX: Incident happens at edge -201089423#1 at time 222 lasting for 2040 seconds, ...
INFO trex_env: Saving metrics to .../results/MAXPRESSURE-tr0-ingolstadt7-7-mplight-wait/metrics_1.csv
```

Episode completed, metrics written, no traceback. Repeated identically for `ingolstadt21`,
`cologne3`, `cologne8` — all four completed cleanly, no lingering SUMO processes afterward
(`pgrep -af sumo` empty). **Conclusion: not a contradiction — round 2 fixed this in commit
`ce3552646`; the round-3 brief's description of the bug matches the *pre-round-2* state,
not what's actually on this branch now.** Marking resolved based on this re-run, not on the
files merely existing.

**One new, unrelated issue surfaced by this re-run, not previously caught**: the first
`ingolstadt7` incident logged `pos=-0.1202005682457532` (a **negative** position) and
`lasting for 0 seconds`. `Initializer.random_pos()` computes
`np.random.uniform(10, edge_length - 10)`; NumPy's behavior is officially undefined when
`high < low`, which happens whenever `edge_length < 20` (a short edge/connector). A 0-second
duration is separately valid (the exponential draw can legitimately round to 0), but the
negative position indicates `random_pos()` has no guard for short edges. 🚩 **Flagged, not
fixed** — this is incident-sampling scientific logic, out of round 3's explicit scope (see
"Do not touch" list), and fixing it requires a modeling decision (skip short edges? clamp
the position? exclude edges below a minimum length from candidate selection?) that should
go through the same owner-decision process as the other sampling-formula items below.

> **Update (round 4):** fixed — short edges (`<20m`) are now excluded from the incident
> edge-candidate pool. See §4.4.

### 3.2 CAV4 teleport exemption — now completed for all 8 networks

Round 2 completed this for 6 of 8 networks (`grid4x4`, `arterial4x4`, `cologne3`,
`cologne8`, `ingolstadt7`, `ingolstadt21`); `cologne1`/`ingolstadt1` had **no `.add.xml` at
all**, so `--strategy 2` would have failed outright there (missing additional-file), not
merely hit the CAV4 issue. Completed now:

- Created `environments/cologne1/cologne1.add.xml` and
  `environments/ingolstadt1/ingolstadt1.add.xml`, copied verbatim from `ingolstadt21`'s
  (the original, working definition) rather than reintroducing anything from the unrelated,
  still-disabled legacy vType-distribution block present in every file — per instruction,
  no new attributes invented.
- Confirmed via `main.py`'s own `--map` argparse choices that these 8 networks
  (`grid4x4`, `arterial4x4`, `cologne1`, `cologne3`, `cologne8`, `ingolstadt1`,
  `ingolstadt7`, `ingolstadt21`) are the complete "supported" set — `map_config.py` also
  defines `arterial5x5`/`turin5`, but those aren't reachable via `--map` at all (not in
  `main.py`'s `choices=[...]`), so they're out of scope for this completion, consistent
  with round 1's finding that they're unused RESCO-inherited leftovers.
- `tests/test_incident_vtypes.py` extended from 6 to all 8 networks (static XML check, no
  SUMO needed) — 8/8 passing.
- **New**: `tests/test_cav4_live.py` — a live-SUMO test that goes further than "doesn't
  crash": it runs one full incident episode per network and asserts the exemption path was
  actually *exercised* (captures the `"exempting queued vehicle"` DEBUG log record), not
  merely that it wasn't triggered. Actual run, this session, all 8 networks,
  `--eps 2 --verbose`:

  | Network | Exemption events (2 episodes) | `not known` errors |
  |---|---|---|
  | grid4x4 | 323 | 0 |
  | arterial4x4 | 366 | 0 |
  | cologne1 | 70 | 0 |
  | cologne3 | 746 | 0 |
  | cologne8 | 293 | 0 |
  | ingolstadt1 | 144 | 0 |
  | ingolstadt7 | 129 | 0 |
  | ingolstadt21 | 105 | 0 |

  `tests/test_cav4_live.py` re-run standalone (1 episode/network via pytest): **8/8
  passing**, each with at least one real exemption event captured, zero "not known" errors.
- **Unrelated discovery, included rather than discarded**: at the start of this round, `git
  status` showed *uncommitted* changes to `grid4x4`/`arterial4x4`/`cologne3`/`cologne8`/
  `ingolstadt7`'s `.add.xml` — not made by any prior round of this audit — adding
  `timeToTeleport="-1"` to the `IC` vType (the incident-blocking dummy vehicle itself, not
  `CAV4`). This looks like the repo owner's own in-progress edit (it exactly matches
  `ingolstadt21.add.xml`'s pre-existing `IC` treatment) and makes sense on its own merits —
  without it, SUMO's global teleport timeout could theoretically remove the very vehicle
  simulating the blockage. Left in place (not reverted or overwritten) and folded into the
  two new `cologne1`/`ingolstadt1` files for consistency, since they were built from the
  same `ingolstadt21` template. Flagged here for visibility since it wasn't something any
  audit round asked for or produced.

### 3.3 `--strategy 3` fail-fast — already done in round 2, now has a test

The behavior itself (`main.py` raising `NotImplementedError` for `--strategy 3` before any
environment is constructed, plus updated `--help` text) was implemented in round 2, commit
`3cdad97e3`, and re-confirmed live this round:

```
$ python3 main.py --agent MAXPRESSURE --map grid4x4 --eps 1 --strategy 3 --libsumo True
...
NotImplementedError: --strategy 3 (curriculum) is planned but not yet implemented. Use --strategy 1 (base) or --strategy 2 (incident).
```

What round 2 skipped was the **unit test** for it — `tests/test_main_cli.py` (new):
asserts `--strategy 3` raises `NotImplementedError` (not `TypeError`) with a message
matching `--strategy 3`, and that `--strategy 1`/`2` still parse normally. No SUMO required
(the check runs before any environment is constructed). 2/2 passing.

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
4. **Initially preserved per-mode, then removed entirely per repo-owner decision — the
   global `--time-to-teleport` flag.** Pre-merge `BaseEnv` unconditionally passed
   `--time-to-teleport -1` (global teleport disabled for every vehicle); pre-merge
   `IncidentEnv` did not (superseded by the per-vehicle `CAV4` teleport exemption mechanism
   added earlier in this audit). `TrexEnv` initially kept this conditional on
   `enable_incidents`, matching each original exactly, on the reasoning that it materially
   affects simulation dynamics and so shouldn't be silently converged. The repo owner then
   clarified: since `CAV4`'s per-vehicle exemption already covers the case a blanket
   override was for (a vehicle genuinely stuck behind a blocked lane), the global flag in the
   "incidents off" path was only ever inherited legacy behavior, not a deliberate
   requirement — so `TrexEnv._seed_args()` now passes neither `--seed`-adjacent teleport
   flag in either mode. Re-verified live: `tests/test_env_unification.py` still passes 3/3
   after this change (no vehicle in the test's short 3-step episode plausibly hits SUMO's
   default ~300s teleport timeout either way) — confirmed empirically, not just assumed. A
   materially longer or more congested episode could in principle show a difference between
   pre-merge `BaseEnv` and `TrexEnv(incident_config=None)` here; that's now an accepted,
   deliberate behavior change, not a bug to chase.
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

## Needs owner decision (round 2) — RESOLVED round 5, see Part B above

> **Update (round 5):** all three of these were confirmed against the manuscript text
> directly by the repo owner and fixed — B1 (`end_time − 1200`), B2 (`warmup=100`), B3
> (`slow_zone_speed=2.2352`). No longer open; kept below for historical context only. See
> the "Round 5 — Part B" section near the top of this report for what changed and the
> live re-verification evidence.

These three items were **deliberately left unresolved** at the time — each was a numeric
discrepancy between the running code and the paper/brief's stated value, in scientifically
load-bearing sampling/behavior code. Only someone with the manuscript in hand (or who made
the recent edit in the `slow_zone_speed` case) could say which side is authoritative; this
audit did not guess. Full context for each is in the linked section below.

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

  > **Update (round 5, B3):** resolved — confirmed against the manuscript text directly by
  > the repo owner ("a conservative reduced speed of 5 mph (approximately 8 km/h)",
  > Section 2.4.2). `slow_zone_speed` is now `2.2352` m/s, the exact 5 mph conversion.

### 2.4 Incident sampling (Section 2.3)

`T_REX.py::Initializer` (lines 84–255).

| Parameter | Paper | Code | Verdict |
|---|---|---|---|
| Blocked lane count | uniform | `random_lanes`: `np.random.randint(1, n_lanes+1)` (level 2) / `randint(1, n_lanes)` (level 1), lanes taken from either end, not fully random subset (commented-out fully-random alternative at `T_REX.py:225-226`) | ✅ Uniform count sampling matches; lane *contiguity* (blocking from one end rather than an arbitrary subset) is a modeling choice not contradicted by the brief's "uniform lane count" description |
| Position `U(10, x_e − 10)` | `random_pos`: `np.random.uniform(10, edge_length - 10)` (`T_REX.py:233`) | ✅ **Exact match** |
| Duration `~Exp(0.029)` | `random_duration`: `np.rint(np.random.exponential(1/0.029)).astype(int)*60` (`T_REX.py:249`) — note `numpy`'s `exponential(scale)` takes `scale=1/rate`, and the result is in **minutes**, multiplied by 60 for seconds | ✅ Matches `Exp(0.029)` under the standard rate parameterization, assuming the paper's rate is per-minute (consistent with the `*60` conversion and with realistic incident durations) |
| Start time `U(t_warmup, t_end − 1200)` | `random_time`: `np.rint(np.random.uniform(self.warm_up_time, self.end_time - 1200)).astype(int)` (`T_REX.py`) | ✅ **Fixed round 5 (B1)** — was `end_time - 500`, confirmed against the manuscript directly by the repo owner and corrected to `end_time - 1200`, matching Section 2.3 exactly. (Historical note: this row previously read "Mismatch, flagged not fixed" pending that confirmation.) |
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

Originally 🚩 **flagged for human review, not changed** — fixing this changes what "the same
seed" reproduces, exactly the kind of scientific-logic change ground rule 3 says must be
flagged rather than silently altered.

> **Update (round 5, B4):** the repo owner confirmed via the manuscript (Section 3.4,
> "averaged over five random seeds") that full-pipeline seed control is part of the stated
> methodology, not optional — applied as a confirmed, deliberate methodology change (not a
> silent alteration): SUMO now gets a deterministic `--seed` (rounds 1-2's work, already
> live), and the RL agents' exploration policies now draw from an independent
> `np.random.default_rng()` `Generator` instead of the global RNG `Initializer.random()`
> reseeds every episode. See "Round 5 — Part B4" near the top of this report for the full
> writeup, the honest scoping caveat (IDQN's epsilon-vs-explore coin flip still lives inside
> `pfrl`'s own library internals, not reachable from this codebase), and the reproducibility
> proof (same seed → identical draws; different seeds → diverge; global reseeding doesn't
> perturb the independent Generator — all three verified against the real production code
> paths, not simulated).

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

> **Update (round 2/3):** the crash itself is fixed — `main.py` now raises a clear
> `NotImplementedError` for `--strategy 3` before any environment is constructed, and the
> `--help` text no longer lists it as a working option. The curriculum feature itself is
> still not implemented (that part of this finding stands). See §3.3.

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

   > **Update (round 2/3):** the repo owner supplied the four CSVs; round 2 committed them
   > (`ce3552646`) and fixed the CWD-dependent path resolution described above at the same
   > time. Round 3 re-ran the exact failing command from this item live and confirmed it
   > now completes on all four real-world networks. See §3.1 for the re-run evidence —
   > resolved based on that, not on the files merely being present.
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

   > **Update (round 3):** the repo owner directed copying `ingolstadt21`'s definition
   > verbatim (no new attributes invented) to the remaining networks, including two
   > (`cologne1`, `ingolstadt1`) that needed a new `.add.xml` created from scratch. Done and
   > live-verified on all 8 networks — see §3.2 for the per-network exemption-event counts.

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
| Critical | `Ing21_prob.csv`/`Ing7_prob.csv`/`Col3_prob.csv`/`Col8_prob.csv` missing — incident scenario cannot run at all on any real-world network | ✅ Fixed round 2, re-verified round 3 by re-running the exact failing command live (§3.1) — data restored, path resolution made CWD-independent |
| Bug | `IncidentEnv._build_sumo_command` route branch referenced a nonexistent `vtypes.add.xml` instead of the already-computed `self.additional` — broke grid4x4/arterial4x4 incident runs entirely | ✅ Fixed |
| Bug | `CAV4` teleport-exemption vType only active in `ingolstadt21.add.xml` elsewhere; `cologne1`/`ingolstadt1` had no `.add.xml` at all | ✅ Fixed round 3 for all 8 networks (§3.2) — live-verified on every network with real exemption events, not just absence of a crash |
| Bug | `--strategy 3` ("curriculum") documented but unimplemented — crashed with `TypeError: range(None)` | ✅ Fixed round 2 (fails fast with `NotImplementedError`), test added round 3 (§3.3) |
| Bug | `Initializer.random_pos()` can return a negative position on short edges (`edge_length < 20`, `np.random.uniform(10, edge_length-10)` with `high < low` is undefined) — found live during round 3's CSV re-verification (§3.1) | ✅ Fixed round 4 (§4.4) — short edges excluded from the candidate pool; exact threshold flagged as an overridable default |
| Correctness note | Table B1's 5 incident categories aren't a selectable code parameter; single generic blockage mechanism | 🚩 Flagged, README describes actual configurability only |
| Critical | No `.gitignore`; 1.3GB+ of generated route files and `.pyc` files tracked in git (6.4GB `.git`) | ✅ Fixed (gitignore + pycache removal) / 🚩 Flagged (arterial4x4 route files, size) |
| Critical | Per-network learning rate defaults not implemented; CLI default silently overrides paper-correct values | ✅ Fixed (added hyperparameter config) |
| Bug | Duplicate `get_arcs_cost` definition, first is dead code | ✅ Fixed |
| Bug | `MPLight.__init__` accepts `lr` but hardcodes `0.005` for the underlying `DQNAgent`, ignoring it | ✅ Fixed |
| Bug | `main.py::run_episode` discards its `obs` param, silently breaking `--repeat` seed replay | ✅ Fixed |
| Bug | TraCI/SUMO lifecycle has no exception safety → subprocess leak on error | ✅ Fixed |
| Bug | Incident start-time upper bound `end_time-500` vs paper's `end_time-1200` | ✅ Fixed round 5 (B1) — manuscript-confirmed |
| Bug | `slow_zone_speed=1.39` contradicted its own comment (`13.8`) and the paper's ~8km/h figure | ✅ Fixed round 5 (B3) — manuscript-confirmed, now `2.2352` |
| Bug | SUMO never seeded (`--random` always); RL exploration RNG entangled with incident-seed reseeding | ✅ Fixed round 5 (B4) — SUMO `--seed` (rounds 1-2) + independent exploration RNG (round 5), manuscript-confirmed methodology |
| Bug | `warmup=0` for every network | ✅ Fixed round 5 (B2) — manuscript-confirmed, now `100` |
| Style | Pervasive `print()` debugging instead of `logging` | ✅ Fixed |
| Style | `assert` used for runtime validation in sampling code | 🚩 Flagged |
| Cleanup | Large commented-out dead-code blocks in `T_REX.py` | 🚩 Flagged |
| Cleanup | `base_env.py`/`incident_env.py` ~90% duplicated boilerplate | 🚩 Flagged (not refactored — too risky to auto-apply) |
| Cleanup | `readXML.py` hardcoded path assumptions, dead/unused script | 🚩 Flagged |
| Hygiene | No LICENSE, tests, CI, CITATION.cff, CONTRIBUTING.md, pinned deps | ✅ Fixed (Phase 5/6) — LICENSE choice needs human confirmation |
| Hygiene | `resco_benchmark` imported but not in `requirements.txt` | ✅ Fixed (documented/pinned) |
