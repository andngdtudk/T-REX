"""Unit tests for the ICM (Information Comply Model) rerouting-decision math in
T_REX.py::Deployment: the binomial-logit rerouting probability
    P = 1 / (1 + exp(-(beta_0 + beta_gain*delta_p - beta_loss*delta_w)))
and its two inputs, expected gain (delta_p, cosine-similarity based) and avoided
loss (delta_w, relative-cost based). See Appendix A of the T-REX paper.

Deployment's real constructor needs a live SUMO/TraCI connection and a .net.xml
file, so these tests build a bare instance via __new__ and set only the
attributes each method under test actually reads, rather than going through
__init__. `add_information_noise` (which injects driver-type-dependent Gaussian
noise) is monkeypatched to the identity function so the surrounding math is
tested deterministically; its own sampling behavior is out of scope here.
"""
import math

import numpy as np
import pytest

from T_REX import Deployment


@pytest.fixture
def deployment():
    dep = Deployment.__new__(Deployment)
    dep.add_information_noise = lambda values, noise_std=0.1: np.asarray(values, dtype=np.float64)
    return dep


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def test_reroute_model_matches_manual_logit(deployment):
    # actual == typical -> cosine similarity 1 -> delta_p = 0
    actual = typical = [0.5, 0.5]
    # same probs dotted with any arc_costs give equal totals -> delta_w = 0
    arc_costs = [10.0, 20.0]
    beta_0, beta_gain, beta_loss = -5, 2.5, 2.5

    prob = deployment.reroute_model(actual, typical, arc_costs, beta_0, beta_gain, beta_loss)

    assert math.isclose(prob, sigmoid(beta_0), rel_tol=1e-9)


def test_reroute_model_returns_zero_when_no_downstream_edges(deployment):
    # get_actual_probs/get_typical_probs return [] when a vehicle has no
    # downstream edges (e.g. at a dead end) -- reroute_model must not raise.
    prob = deployment.reroute_model([], [], [], -5, 2.5, 2.5)
    assert prob == 0.0


def test_expected_gain_is_zero_for_identical_distributions(deployment):
    delta_p = deployment.calculate_expected_gain([0.3, 0.7], [0.3, 0.7])
    assert math.isclose(delta_p, 0.0, abs_tol=1e-9)


def test_expected_gain_is_one_for_orthogonal_distributions(deployment):
    # cosine similarity of orthogonal vectors is 0 -> delta_p = 1 - 0 = 1
    delta_p = deployment.calculate_expected_gain([1.0, 0.0], [0.0, 1.0])
    assert math.isclose(delta_p, 1.0, abs_tol=1e-9)


def test_avoided_loss_is_zero_for_identical_distributions(deployment):
    delta_w = deployment.calculate_avoided_loss([0.4, 0.6], [0.4, 0.6], [10.0, 5.0])
    assert math.isclose(delta_w, 0.0, abs_tol=1e-9)


def test_avoided_loss_positive_when_actual_avoids_the_costly_arc(deployment):
    # typical prefers the expensive arc; actual avoids it -> positive avoided loss
    typical = [0.9, 0.1]
    actual = [0.1, 0.9]
    arc_costs = [100.0, 1.0]  # first arc is far more costly
    delta_w = deployment.calculate_avoided_loss(actual, typical, arc_costs)
    assert delta_w > 0


def test_reroute_probability_increases_with_expected_gain(deployment):
    # Higher delta_p (bigger mismatch between actual and typical routing) should
    # raise the rerouting probability, holding avoided loss fixed at 0.
    arc_costs = [10.0, 10.0]  # equal costs -> delta_w stays 0 regardless of probs
    low_gain = deployment.reroute_model([0.5, 0.5], [0.5, 0.5], arc_costs, -5, 2.5, 2.5)
    high_gain = deployment.reroute_model([1.0, 0.0], [0.0, 1.0], arc_costs, -5, 2.5, 2.5)
    assert high_gain > low_gain
