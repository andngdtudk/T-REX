"""Unit tests for the SUMO --seed/--random selection in BaseEnv/IncidentEnv.

Both envs' real constructors need a live SUMO connection, so these tests build
bare instances via __new__ and set only sumo_seed/run, which is all
_seed_args() reads.
"""
import pytest

from base_env import BaseEnv
from incident_env import IncidentEnv


@pytest.mark.parametrize("env_cls", [BaseEnv, IncidentEnv])
def test_random_when_no_seed_given(env_cls):
    env = env_cls.__new__(env_cls)
    env.sumo_seed = None
    env.run = 3
    assert env._seed_args() == ['--random']


@pytest.mark.parametrize("env_cls", [BaseEnv, IncidentEnv])
def test_seed_offset_by_episode_number(env_cls):
    env = env_cls.__new__(env_cls)
    env.sumo_seed = 42
    env.run = 0
    assert env._seed_args() == ['--seed', '42']

    env.run = 5
    assert env._seed_args() == ['--seed', '47']


@pytest.mark.parametrize("env_cls", [BaseEnv, IncidentEnv])
def test_same_seed_and_run_are_deterministic(env_cls):
    env_a = env_cls.__new__(env_cls)
    env_a.sumo_seed, env_a.run = 7, 2
    env_b = env_cls.__new__(env_cls)
    env_b.sumo_seed, env_b.run = 7, 2

    assert env_a._seed_args() == env_b._seed_args()
