"""Unit tests for T_REX.py::Initializer's random incident-parameter sampling
(Section 2.3 of the T-REX paper): blocked-lane count, position, start time,
and duration.

Initializer's real constructor needs a live SUMO/TraCI connection and reads a
.net.xml file via sumolib, so these tests build a bare instance via __new__
and set only the attributes each sampling method actually reads, then
monkeypatch the `traci` calls those methods make (getLaneNumber/getLength)
rather than spinning up SUMO.

random_time()'s upper bound is `end_time - 1200`, confirmed against the manuscript
text directly by the repo owner (round 5); see AUDIT_REPORT.md.
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


def test_random_time_within_paper_bounds(initializer):
    for _ in range(200):
        initializer.random_time()
        assert initializer.warm_up_time <= initializer.start_time <= initializer.end_time - 1200
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


class _FakeDownstreamEdge:
    """random_edge() calls .getID() on each key of getOutgoing()'s dict (sumolib
    Edge objects, not strings) -- provide that."""

    def getID(self):
        return "downstream"


class _FakeEdge:
    """Minimal stand-in for a sumolib net Edge -- only what random_edge()/
    weighted_random_edge() call: getOutgoing()."""

    def __init__(self, has_outgoing=True):
        self._has_outgoing = has_outgoing

    def getOutgoing(self):
        return {_FakeDownstreamEdge(): None} if self._has_outgoing else {}


class _FakeNet:
    """Minimal stand-in for a sumolib net Net -- only getEdge(edge_id)."""

    def __init__(self, edges):
        self._edges = edges

    def getEdge(self, edge_id):
        return self._edges[edge_id]


def _patch_edge_lengths(lengths):
    """Return a context manager patching traci.lane.getLength to answer per-edge,
    keyed the same way the code under test looks it up: f'{edge}_0'."""
    def fake_get_length(lane_id):
        edge_id = lane_id.rsplit("_", 1)[0]
        return lengths[edge_id]
    return patch.object(T_REX.traci.lane, "getLength", side_effect=fake_get_length)


def test_random_edge_excludes_edges_shorter_than_min_length(initializer):
    # short_edge is 15m (< MIN_INCIDENT_EDGE_LENGTH=20) and must never be selected;
    # long_edge is 200m and must be the only edge ever selected here.
    initializer.net = _FakeNet({"short_edge": _FakeEdge(), "long_edge": _FakeEdge()})
    lengths = {"short_edge": 15.0, "long_edge": 200.0}

    with patch.object(T_REX.traci.edge, "getIDList", return_value=["short_edge", "long_edge"]), \
         _patch_edge_lengths(lengths):
        selected = set()
        for _ in range(100):
            initializer.random_edge()
            selected.add(initializer.edge)

    assert selected == {"long_edge"}


def test_random_edge_boundary_length_is_included(initializer):
    # An edge of exactly MIN_INCIDENT_EDGE_LENGTH (20m) leaves zero room, not negative
    # room -- U(10, 10) is well-defined (numpy returns 10.0 deterministically when
    # low == high) -- so it should be eligible, not excluded.
    initializer.net = _FakeNet({"boundary_edge": _FakeEdge()})
    lengths = {"boundary_edge": T_REX.Initializer.MIN_INCIDENT_EDGE_LENGTH}

    with patch.object(T_REX.traci.edge, "getIDList", return_value=["boundary_edge"]), \
         _patch_edge_lengths(lengths):
        initializer.random_edge()

    assert initializer.edge == "boundary_edge"


def test_weighted_random_edge_excludes_edges_shorter_than_min_length(initializer):
    initializer.net = _FakeNet({"short_edge": _FakeEdge(), "long_edge": _FakeEdge()})
    initializer.edge_probabilities = {"short_edge": 0.9, "long_edge": 0.1}
    lengths = {"short_edge": 15.0, "long_edge": 200.0}

    with patch.object(T_REX.traci.edge, "getIDList", return_value=["short_edge", "long_edge"]), \
         _patch_edge_lengths(lengths):
        selected = set()
        for _ in range(100):
            initializer.weighted_random_edge()
            selected.add(initializer.edge)

    # short_edge had 90% of the probability mass, but is short -- if it weren't excluded
    # it would dominate the selection. It must never appear.
    assert selected == {"long_edge"}


def test_random_pos_never_negative_end_to_end(initializer):
    # random_edge() -> random_pos() should never produce a negative position, for a mix
    # of edge lengths including some that would have failed pre-fix (short_edge here
    # would have made np.random.uniform(10, 15-10=5) undefined -- low > high).
    initializer.net = _FakeNet({
        "short_edge": _FakeEdge(),
        "boundary_edge": _FakeEdge(),
        "long_edge": _FakeEdge(),
    })
    lengths = {"short_edge": 15.0, "boundary_edge": 20.0, "long_edge": 200.0}

    with patch.object(T_REX.traci.edge, "getIDList",
                       return_value=["short_edge", "boundary_edge", "long_edge"]), \
         _patch_edge_lengths(lengths):
        for _ in range(200):
            initializer.random_edge()
            assert initializer.edge != "short_edge"
            initializer.random_pos()
            assert initializer.pos >= 10
