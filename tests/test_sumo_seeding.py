"""Unit tests for the SUMO --seed/--random and --time-to-teleport selection in
TrexEnv._seed_and_teleport_args().

TrexEnv's real constructor needs a live SUMO connection, so these tests build a
bare instance via __new__ and set only sumo_seed/run/enable_incidents, which is
all _seed_and_teleport_args() reads.
"""
import pytest

from trex_env import TrexEnv


def _bare_env(sumo_seed, run, enable_incidents):
    env = TrexEnv.__new__(TrexEnv)
    env.sumo_seed = sumo_seed
    env.run = run
    env.enable_incidents = enable_incidents
    return env


@pytest.mark.parametrize("enable_incidents", [False, True])
def test_random_when_no_seed_given(enable_incidents):
    env = _bare_env(sumo_seed=None, run=3, enable_incidents=enable_incidents)
    args = env._seed_and_teleport_args()
    assert args[:1] == ['--random']


@pytest.mark.parametrize("enable_incidents", [False, True])
def test_seed_offset_by_episode_number(enable_incidents):
    env = _bare_env(sumo_seed=42, run=0, enable_incidents=enable_incidents)
    assert env._seed_and_teleport_args()[:2] == ['--seed', '42']

    env.run = 5
    assert env._seed_and_teleport_args()[:2] == ['--seed', '47']


@pytest.mark.parametrize("enable_incidents", [False, True])
def test_same_seed_and_run_are_deterministic(enable_incidents):
    env_a = _bare_env(sumo_seed=7, run=2, enable_incidents=enable_incidents)
    env_b = _bare_env(sumo_seed=7, run=2, enable_incidents=enable_incidents)
    assert env_a._seed_and_teleport_args() == env_b._seed_and_teleport_args()


def test_teleport_disabled_globally_only_when_incidents_off():
    off = _bare_env(sumo_seed=1, run=0, enable_incidents=False)
    on = _bare_env(sumo_seed=1, run=0, enable_incidents=True)

    assert '--time-to-teleport' in off._seed_and_teleport_args()
    assert '-1' in off._seed_and_teleport_args()
    assert '--time-to-teleport' not in on._seed_and_teleport_args()
