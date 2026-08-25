"""Unit tests for the SUMO --seed/--random selection in TrexEnv._seed_args().

TrexEnv's real constructor needs a live SUMO connection, so these tests build a
bare instance via __new__ and set only sumo_seed/run, which is all
_seed_args() reads.
"""
import pytest

from trex_env import TrexEnv


def _bare_env(sumo_seed, run):
    env = TrexEnv.__new__(TrexEnv)
    env.sumo_seed = sumo_seed
    env.run = run
    return env


def test_random_when_no_seed_given():
    env = _bare_env(sumo_seed=None, run=3)
    assert env._seed_args() == ['--random']


def test_seed_offset_by_episode_number():
    env = _bare_env(sumo_seed=42, run=0)
    assert env._seed_args() == ['--seed', '42']

    env.run = 5
    assert env._seed_args() == ['--seed', '47']


def test_same_seed_and_run_are_deterministic():
    env_a = _bare_env(sumo_seed=7, run=2)
    env_b = _bare_env(sumo_seed=7, run=2)
    assert env_a._seed_args() == env_b._seed_args()
