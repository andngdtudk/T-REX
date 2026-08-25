"""Unit tests for T_REX.py::Initializer's random incident-parameter sampling
(Section 2.3 of the T-REX paper): blocked-lane count, position, start time,
and duration.

Initializer's real constructor needs a live SUMO/TraCI connection and reads a
.net.xml file via sumolib, so these tests build a bare instance via __new__
and set only the attributes each sampling method actually reads, then
monkeypatch the `traci` calls those methods make (getLaneNumber/getLength)
rather than spinning up SUMO.

Note: random_time()'s upper bound is checked against the CURRENT
implementation (`end_time - 500`), not the paper's stated `end_time - 1200`
(see AUDIT_REPORT.md Section 2.4) -- that mismatch is flagged for human
review, not silently changed, so this test intentionally documents the
as-implemented behavior rather than the as-published one.
"""
from unittest.mock import patch

import numpy as np
import pytest

import T_REX
from T_REX import Initializer


@pytest.fixture
def initializer():
    init = Initializer.__new__(Initializer)
    init.edge = "test_edge"
    init.map_name = "grid4x4"
    init.level = 2
    init.warm_up_time = 100
    init.end_time = 3600
    return init


def test_random_pos_within_paper_bounds(initializer):
    edge_length = 200.0
    with patch.object(T_REX.traci.lane, "getLength", return_value=edge_length):
        for _ in range(200):
            initializer.random_pos()
            assert 10 <= initializer.pos <= edge_length - 10


def test_random_time_within_implemented_bounds(initializer):
    for _ in range(200):
        initializer.random_time()
        assert initializer.warm_up_time <= initializer.start_time <= initializer.end_time - 500
        assert initializer.start_step == initializer.start_time


def test_random_duration_is_nonnegative_multiple_of_60(initializer):
    for _ in range(200):
        initializer.random_duration()
        assert initializer.duration_time >= 0
        assert initializer.duration_time % 60 == 0
        assert initializer.duration_steps == initializer.duration_time


def test_random_duration_mean_matches_exp_0_029(initializer):
    # Exp(0.029) in minutes has mean 1/0.029 ~= 34.48 minutes = ~2069s.
    # Check the sample mean over many draws is in the right ballpark (loose
    # tolerance -- this is a statistical sanity check, not an exact-value test).
    samples = []
    for _ in range(2000):
        initializer.random_duration()
        samples.append(initializer.duration_time)
    mean_seconds = np.mean(samples)
    expected_mean_seconds = (1 / 0.029) * 60
    assert 0.85 * expected_mean_seconds < mean_seconds < 1.15 * expected_mean_seconds


def test_random_lanes_single_lane_blocks_that_lane(initializer):
    with patch.object(T_REX.traci.edge, "getLaneNumber", return_value=1):
        initializer.random_lanes()
        assert initializer.lanes == [0]


def test_random_lanes_ingolstadt21_excludes_pedestrian_lane_0(initializer):
    initializer.map_name = "ingolstadt21"
    with patch.object(T_REX.traci.edge, "getLaneNumber", return_value=3):
        for _ in range(50):
            initializer.random_lanes()
            assert 0 not in initializer.lanes
            assert set(initializer.lanes).issubset({1, 2})


def test_random_lanes_level_2_can_block_all_lanes(initializer):
    initializer.level = 2
    with patch.object(T_REX.traci.edge, "getLaneNumber", return_value=4):
        blocked_counts = set()
        for _ in range(200):
            initializer.random_lanes()
            assert 1 <= len(initializer.lanes) <= 4
            assert set(initializer.lanes).issubset({0, 1, 2, 3})
            blocked_counts.add(len(initializer.lanes))
        # Level 2 allows blocking every lane; confirm the full range is reachable.
        assert 4 in blocked_counts


def test_random_lanes_level_1_never_blocks_all_lanes(initializer):
    initializer.level = 1
    with patch.object(T_REX.traci.edge, "getLaneNumber", return_value=4):
        for _ in range(200):
            initializer.random_lanes()
            assert 1 <= len(initializer.lanes) < 4
