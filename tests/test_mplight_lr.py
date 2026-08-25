"""Regression test for the MPLight lr plumbing bug (see AUDIT_REPORT.md).

MPLight.__init__ used to accept an `lr` parameter but hardcode lr=0.005
when constructing the underlying DQNAgent, silently discarding whatever
the caller passed. This test builds a real MPLight agent (needs torch,
no SUMO) with several different lr values and inspects the actual PyTorch
optimizer's param_groups -- not just that the source line was edited.
"""
import pytest

torch = pytest.importorskip("torch")


def _build_mplight(lr):
    from TREX_comp.agents.mplight import MPLight

    config = {
        'BATCH_SIZE': 32, 'GAMMA': 0.99, 'EPS_START': 1.0, 'EPS_END': 0.0,
        'EPS_DECAY': 220, 'TARGET_UPDATE': 500, 'num_lights': 1, 'load': False,
        'demand_shape': 1, 'steps': 1000,
    }
    return MPLight(config, {}, 'grid4x4', 0, lr=lr)


@pytest.mark.parametrize("lr", [0.005, 0.001, 0.05, 1e-5])
def test_mplight_lr_reaches_optimizer(lr):
    agent = _build_mplight(lr)
    actual_lr = agent.agent.optimizer.param_groups[0]['lr']
    assert actual_lr == lr, (
        f"requested lr={lr} but the underlying optimizer has lr={actual_lr} -- "
        "MPLight is silently ignoring its lr parameter again"
    )


def test_mplight_default_lr_is_0_005():
    agent = _build_mplight(0.005)
    assert agent.agent.optimizer.param_groups[0]['lr'] == 0.005
