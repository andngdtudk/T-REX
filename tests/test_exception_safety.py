"""Regression test for TraCI/SUMO exception safety (see AUDIT_REPORT.md Part A.2).

Before this was fixed, an exception raised between traci.start() and the
matching traci.close() (e.g. a bug in state_fn/reward_fn, a TraCI RPC error)
would leak the SUMO subprocess -- reset()/close() didn't wrap the teardown in
try/finally. This test forces exactly that failure mode (a reward_fn that
raises mid-episode) and confirms env.close() still runs cleanly and the
simulation is genuinely torn down, not left in a broken half-open state.

A companion *live* check (not part of this suite, run manually, see
AUDIT_REPORT.md Part A.2) used libsumo=False so a real 'sumo' OS subprocess
is spawned, and confirmed via pgrep that it's actually gone after close() --
that's the strongest form of this evidence, but isn't repeated here as an
automated test because spawning a real subprocess per CI run is slow/fragile.
This test instead proves the same code path is exception-safe using libsumo
(fast, matches how the rest of this suite runs), via two checks that would
fail if teardown were broken: close() itself must not raise a second,
different error while unwinding, and a second env must be constructable
afterward (proving libsumo's single global simulation slot was actually
released, not left occupied by the failed one).

Requires SUMO/libsumo; skipped otherwise.
"""
import os

import pytest


def _sumo_available():
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not installed")


def _make_env(tmp_path, reward_fn, run_name="excsafety"):
    from TREX_comp import states, rewards as rewards_module
    from TREX_comp.config.map_config import map_configs
    from TREX_comp.config.incident_config import IncidentConfig
    from trex_env import TrexEnv

    map_config = map_configs["ingolstadt7"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return TrexEnv(
        run_name=run_name,
        map_name="ingolstadt7",
        net=os.path.join(repo_root, map_config["net"]),
        state_fn=states.mplight,
        reward_fn=reward_fn,
        route=None,
        step_length=map_config["step_length"],
        yellow_length=map_config["yellow_length"],
        step_ratio=map_config["step_ratio"],
        end_time=map_config["end_time"],
        max_distance=200,
        lights=map_config["lights"],
        gui=False,
        log_dir=str(tmp_path) + os.sep,
        libsumo=True,
        warmup=0,
        run=0,
        incident_config=IncidentConfig(level=2),
        sumo_seed=1,
    )


def test_mid_episode_exception_still_closes_cleanly(tmp_path):
    from TREX_comp import rewards

    call_count = {"n": 0}

    def flaky_reward(signals):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("deliberate mid-episode failure for this test")
        return rewards.wait(signals)

    env = _make_env(tmp_path, flaky_reward)

    raised = None
    try:
        obs = env.reset()
        done = False
        while not done:
            act = {ts: 0 for ts in env.signal_ids}
            obs, rew, done, info = env.step(act)
    except RuntimeError as e:
        raised = e
    finally:
        # This must not itself raise -- that would mean the exception left
        # the connection in a state close()'s own traci.close()/save_metrics
        # try/finally couldn't recover from.
        env.close()

    assert raised is not None, "the deliberately-injected exception should have propagated"
    assert call_count["n"] == 2, "reward_fn should have been called exactly up to the failure point"

    # Proves libsumo's single global simulation slot was actually released,
    # not left occupied by the failed run -- constructing (and fully using)
    # a second, independent env would fail immediately otherwise.
    from TREX_comp import rewards as rewards_module
    env2 = _make_env(tmp_path, rewards_module.wait, run_name="excsafety-followup")
    try:
        obs2 = env2.reset()
        act2 = {ts: 0 for ts in env2.signal_ids}
        env2.step(act2)
    finally:
        env2.close()
