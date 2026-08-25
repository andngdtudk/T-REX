"""Round 7: empirically confirm SharedEpsGreedy's epsilon schedule matches pfrl's
stock LinearDecayEpsilonGreedy exactly, for IDQN's real construction parameters.

Round 6 swapped IDQN's explorer from pfrl's own LinearDecayEpsilonGreedy to the
repo's SharedEpsGreedy (a subclass), justified by matching call signatures against
pfrl.agents.DQN.batch_act. That proves interface compatibility, not that the two
produce the same epsilon value at a given step -- this test confirms the latter
directly, rather than trusting that SharedEpsGreedy not overriding compute_epsilon
is sufficient reasoning on its own.

Uses IDQN's real EPS_START=1.0/EPS_END=0.0 (TREX_comp/config/agent_config.py) and a
realistic decay_steps derived the same way main.py does: int(eps*0.8)*num_steps_eps.
"""
import numpy as np
import pytest
from pfrl import explorers

from TREX_comp.agents.pfrl_dqn import SharedEpsGreedy

EPS_START = 1.0
EPS_END = 0.0
# int(100*0.8) * (3600/10) -- matches main.py's steps derivation for a typical
# --eps 100 --map grid4x4 run (end_time=3600, start_time=0, step_length=10).
DECAY_STEPS = 80 * 360


def _make_pair():
    pfrl_explorer = explorers.LinearDecayEpsilonGreedy(
        EPS_START, EPS_END, DECAY_STEPS, lambda: 0,
    )
    repo_explorer = SharedEpsGreedy(
        EPS_START, EPS_END, DECAY_STEPS, lambda: 0,
        rng=np.random.default_rng(0),
    )
    return pfrl_explorer, repo_explorer


TEST_STEPS = [0, 1, 1000, DECAY_STEPS // 4, DECAY_STEPS // 2,
              (3 * DECAY_STEPS) // 4, DECAY_STEPS - 1, DECAY_STEPS, DECAY_STEPS + 1,
              DECAY_STEPS * 10]


@pytest.mark.parametrize("t", TEST_STEPS)
def test_epsilon_matches_pfrl_exactly(t):
    pfrl_explorer, repo_explorer = _make_pair()
    pfrl_eps = pfrl_explorer.compute_epsilon(t)
    repo_eps = repo_explorer.compute_epsilon(t)
    assert repo_eps == pfrl_eps, f"t={t}: pfrl={pfrl_eps!r} repo={repo_eps!r}"


def test_shared_eps_greedy_does_not_override_compute_epsilon():
    # The two classes literally share the same compute_epsilon code (Python
    # inheritance), not just matching output -- this pins that invariant so a
    # future SharedEpsGreedy change that adds an override would be caught here too.
    assert SharedEpsGreedy.compute_epsilon is explorers.LinearDecayEpsilonGreedy.compute_epsilon


if __name__ == "__main__":
    print(f"{'t':>10} | {'pfrl':>10} | {'repo (SharedEpsGreedy)':>22} | result")
    print("-" * 60)
    for t in TEST_STEPS:
        pfrl_explorer, repo_explorer = _make_pair()
        pfrl_eps = pfrl_explorer.compute_epsilon(t)
        repo_eps = repo_explorer.compute_epsilon(t)
        status = "MATCH" if repo_eps == pfrl_eps else "MISMATCH"
        print(f"{t:>10} | {pfrl_eps:>10.6f} | {repo_eps:>22.6f} | {status}")
