"""Live-SUMO companion to tests/test_incident_vtypes.py.

That test only proves the CAV4 vType is *declared* per network's .add.xml
(static XML check, no SUMO needed). This test proves it actually works at
runtime: for every network, run one full episode with incidents enabled and
assert at least one vehicle was genuinely exempted from SUMO's teleport
timeout (Deployment.manage_incident_queue_teleport_exemption logs "exempting
queued vehicle" at DEBUG level exactly when this happens -- see T_REX.py).
A network that merely doesn't crash but never exercises the exemption path
(e.g. because traffic is too sparse for anything to ever queue) would pass a
crash-only check while silently proving nothing; asserting the log event
actually fired closes that gap.

grid4x4/arterial4x4 need their route-file zips decompressed first (see
README "Installation") and are skipped individually if that hasn't been
done, rather than skipping the whole module -- the .sumocfg-based networks
don't have that dependency and should still run.

Requires a working SUMO installation; skipped entirely otherwise. Each
network's full episode takes on the order of 10-60 seconds real time
(TRACI dominates, not this test), so the whole module is opt-in for CI,
matching test_env_unification.py.
"""
import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# All 8 networks reachable via main.py's --map (matches
# TREX_comp/config/map_config.py's keys that also appear in main.py's
# argparse choices -- arterial5x5/turin5 are in map_config.py but not
# selectable via --map and are out of scope).
ALL_NETWORKS = [
    "grid4x4",
    "arterial4x4",
    "cologne1",
    "cologne3",
    "cologne8",
    "ingolstadt1",
    "ingolstadt7",
    "ingolstadt21",
]

ROUTE_BASED_NETWORKS = {"grid4x4", "arterial4x4"}


def _sumo_available():
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not installed")


@pytest.fixture
def caplog_debug(caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger="T_REX")
    return caplog


@pytest.mark.parametrize("network", ALL_NETWORKS)
def test_cav4_exemption_fires_on_every_network(network, caplog_debug, tmp_path):
    if network in ROUTE_BASED_NETWORKS:
        route_dir = REPO_ROOT / "environments" / network / network
        if not route_dir.is_dir():
            pytest.skip(f"{network} traffic-flow files not decompressed (see README Installation)")

    # sumo_seed below only controls SUMO's own internal --seed (vehicle-level
    # stochasticity); WHICH incident gets sampled (edge/lane/position/duration) is
    # drawn from the global numpy RNG state Initializer.random() reseeds itself from
    # (T_REX.py) -- without pinning that too, this test's outcome depends on test
    # execution order/how much prior global RNG state other tests consumed, not just
    # on network identity. Found the hard way: this test passed in isolation but
    # failed as part of the full suite (ingolstadt1's randomly-sampled incident
    # happened not to queue anyone within one episode that run). Match main.py's own
    # np.random.seed(args.seed) pattern so this is genuinely deterministic.
    import numpy as np
    np.random.seed(1)

    from TREX_comp import states, rewards
    from TREX_comp.config.map_config import map_configs
    from TREX_comp.config.incident_config import IncidentConfig
    from trex_env import TrexEnv

    map_config = map_configs[network]
    route = os.path.join(str(REPO_ROOT), map_config["route"]) if map_config.get("route") else None

    env = TrexEnv(
        run_name=f"cav4live-{network}",
        map_name=network,
        net=os.path.join(str(REPO_ROOT), map_config["net"]),
        state_fn=states.mplight,
        reward_fn=rewards.wait,
        route=route,
        step_length=map_config["step_length"],
        yellow_length=map_config["yellow_length"],
        step_ratio=map_config["step_ratio"],
        end_time=map_config["end_time"],
        max_distance=200,
        lights=map_config["lights"],
        gui=False,
        log_dir=str(tmp_path) + os.sep,
        libsumo=True,
        warmup=map_config["warmup"],
        run=0,
        incident_config=IncidentConfig(level=2),
        sumo_seed=1,
    )

    try:
        obs = env.reset()
        done = False
        while not done:
            act = {ts: 0 for ts in env.signal_ids}
            obs, rew, done, info = env.step(act)
    finally:
        env.close()

    exemption_events = [
        r for r in caplog_debug.records
        if "exempting queued vehicle" in r.getMessage()
    ]
    not_known_errors = [
        r for r in caplog_debug.records
        if "not known" in r.getMessage()
    ]

    assert not not_known_errors, (
        f"{network}: CAV4 (or another vType) was rejected as unknown during "
        f"the run: {[r.getMessage() for r in not_known_errors]}"
    )
    assert exemption_events, (
        f"{network}: ran a full incident episode without a single vehicle "
        "queuing behind a blocked lane -- the CAV4 exemption path was never "
        "actually exercised, so this run doesn't prove it works here. "
        "(Empirically, every one of these 8 networks produced dozens to "
        "hundreds of exemption events in a 2-episode manual run during this "
        "audit -- see AUDIT_REPORT.md -- so an empty result on a single "
        "episode is more likely a real regression than bad luck.)"
    )
