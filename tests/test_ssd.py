"""Unit tests for the AASHTO stopping-sight-distance (SSD) calculation.

T_REX.py::Deployment.calculate_ssd implements
    SSD = v * t_perception_reaction + v^2 / (2 * a_deceleration)
with t=2.5s, a=3.4 m/s^2 (Section 2.4.2 / Appendix B of the T-REX paper).
"""
import math

from T_REX import Deployment


def test_ssd_uses_paper_aashto_constants():
    assert Deployment.SSD_PERCEPTION_REACTION_TIME == 2.5
    assert Deployment.SSD_DECELERATION == 3.4


def test_ssd_zero_speed_is_zero():
    assert Deployment.calculate_ssd(0) == 0.0


def test_ssd_matches_manual_formula():
    speed = 13.9  # ~50 km/h
    expected = speed * 2.5 + (speed ** 2) / (2 * 3.4)
    assert math.isclose(Deployment.calculate_ssd(speed), expected, rel_tol=1e-9)


def test_ssd_increases_with_speed():
    # Both terms of the SSD formula are monotonically increasing in v for v >= 0,
    # so SSD itself must be monotonically increasing.
    speeds = [0, 5, 10, 15, 20, 27.8]
    ssds = [Deployment.calculate_ssd(v) for v in speeds]
    assert ssds == sorted(ssds)
    assert len(set(ssds)) == len(ssds)  # strictly increasing, not just non-decreasing
