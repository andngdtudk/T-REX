"""Regression test for the run_episode obs-discarding bug (see AUDIT_REPORT.md).

main.py::run_episode(env, agent, obs=None) used to call env.reset()
unconditionally, discarding whatever pre-seeded obs its caller already
computed via env.reset(pre_seed=[...]) -- silently breaking --repeat's
fixed-incident-seed replay (main.py::run_incident_scenario). A full CLI-level
reproduction of that bug (train with --repeat, then test --load --repeat with
a *different* top-level --seed, and diff the incident logs) was run live for
AUDIT_REPORT.md; this test covers the same mechanism faster and without file
I/O, by checking a very direct symptom of the bug: run_episode must NOT call
env.reset() a second time when it's already been given a valid obs (a second
reset() is externally observable as env.run incrementing again).

Requires SUMO; skipped otherwise.
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


class _NoopAgent:
    """Just enough of an agent interface for run_episode's loop."""

    def act(self, obs):
        return {}

    def observe(self, obs, rew, done, info):
        pass


def test_run_episode_does_not_reset_again_when_given_obs(tmp_path):
    import main
    from TREX_comp import states, rewards
    from TREX_comp.config.map_config import map_configs
    from TREX_comp.config.incident_config import IncidentConfig
    from trex_env import TrexEnv

    map_config = map_configs["ingolstadt7"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env = TrexEnv(
        run_name="repeatreplay",
        map_name="ingolstadt7",
        net=os.path.join(repo_root, map_config["net"]),
        state_fn=states.mplight,
        reward_fn=rewards.wait,
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

    try:
        # Reset with an explicit pre_seed, the same mechanism --repeat's
        # replay path uses (main.py: env.reset(pre_seed=[seed_ic1, seed_ic2])).
        obs = env.reset(pre_seed=[111, 222])
        run_after_reset = env.run
        seed_ic1_after_reset = env.seed_ic1
        seed_ic2_after_reset = env.seed_ic2
        assert seed_ic1_after_reset == 111
        assert seed_ic2_after_reset == 222

        # This is the exact call pattern run_incident_scenario uses for
        # replay: run_episode(env, agent, obs) with a non-None obs already
        # in hand from the pre-seeded reset above.
        main.run_episode(env, _NoopAgent(), obs)

        # If run_episode had (bug-era) called env.reset() again internally,
        # env.run would have incremented a second time, and the pre_seed
        # above would have been silently replaced by a fresh, unseeded one.
        assert env.run == run_after_reset, (
            "run_episode appears to have reset the environment again "
            "internally, despite being given a valid pre-seeded obs -- this "
            "is exactly the bug that broke --repeat's incident-seed replay"
        )
        assert env.seed_ic1 == seed_ic1_after_reset
        assert env.seed_ic2 == seed_ic2_after_reset
    finally:
        env.close()
