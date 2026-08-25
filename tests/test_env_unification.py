"""Mandatory regression test for the base_env/incident_env -> trex_env merge.

Runs the pre-merge reference implementations (tests/reference_impl/, exact
snapshots of base_env.py/incident_env.py taken immediately before this
refactor) and the new unified TrexEnv side by side, same fixed seed, same
short episode, and asserts their outputs are identical:

- "off" comparison: pre-merge BaseEnv vs. TrexEnv(incident_config=None)
- "on" comparison: pre-merge IncidentEnv vs. TrexEnv(incident_config=IncidentConfig(level=2))

Both comparisons run on ingolstadt7 (a .sumocfg-based network, route=None)
rather than a route-based network like grid4x4/arterial4x4: diffing the two
pre-merge files surfaced a real, pre-existing bug where pre-merge BaseEnv's
route-file path construction did not match pre-merge IncidentEnv's/main.py's
(see trex_env.py's class docstring and AUDIT_REPORT.md) -- pre-merge BaseEnv
cannot actually run on grid4x4/arterial4x4 at all given how those networks
are documented to be decompressed, so no "old" baseline exists there to
compare against. TrexEnv fixes that bug for both modes rather than
reproducing it, which is itself verified by test_grid4x4_base_scenario_works
below (a regression test for the fix, since there is no old behavior to
match here -- only that it no longer crashes).

Requires a working SUMO installation; skipped otherwise.
"""
import importlib.util
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "tests" / "reference_impl"


def _sumo_available():
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401
        return shutil.which("sumo") is not None or shutil.which("sumo.exe") is not None or True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _sumo_available(), reason="SUMO/traci not installed")


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_episode(env_ctor_kwargs, env_class, seed, steps, pre_seed=None):
    """Construct one env, run `steps` step() calls with a constant action, close it.

    Returns (connection_name, log_dir) so the caller can read back the
    metrics_1.csv/tripinfo_1.xml files it wrote.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = env_class(**env_ctor_kwargs)
    if pre_seed is not None:
        obs = env.reset(pre_seed=pre_seed)
    else:
        obs = env.reset()

    for _ in range(steps):
        act = {ts: 0 for ts in env.signal_ids}
        obs, rew, done, info = env.step(act)

    connection_name = env.connection_name
    log_dir = env.log_dir
    env.close()
    return connection_name, log_dir


def _read_metrics(log_dir, connection_name, run=1, normalize_trailing_comma=False):
    """Read metrics_<run>.csv.

    normalize_trailing_comma: pre-merge BaseEnv.save_metrics built each line by
    manually concatenating `str(value) + ', '` for every field including the
    last, leaving a stray trailing ", " before the newline
    (tests/reference_impl/base_env_pre_merge.py:257-259). Pre-merge
    IncidentEnv.save_metrics used `', '.join(...)` instead, which doesn't have
    that trailing separator. TrexEnv unifies both modes onto the cleaner
    IncidentEnv format (same data, no stray trailing comma) rather than
    special-casing one extra byte per line depending on incident_config --
    see AUDIT_REPORT.md. Set this when reading pre-merge BaseEnv's output so
    the comparison isn't polluted by that one cosmetic byte; every other test
    in this file (including the "incidents on" comparison) does NOT need this
    and asserts byte-for-byte identity with no normalization.
    """
    path = os.path.join(log_dir, connection_name, f"metrics_{run}.csv")
    with open(path) as f:
        content = f.read()
    if normalize_trailing_comma:
        content = content.replace(", \n", "\n")
    return content


def _read_tripinfo_vehicle_ids_and_durations(log_dir, connection_name, run=1):
    """Extract (id, duration) pairs from tripinfo_<run>.xml, ignoring any
    header/comment lines that could vary between two otherwise-identical runs
    (e.g. a generation timestamp SUMO may emit at the top of the file)."""
    import xml.etree.ElementTree as ET

    path = os.path.join(log_dir, connection_name, f"tripinfo_{run}.xml")
    root = ET.parse(path).getroot()
    return sorted(
        (trip.get("id"), trip.get("duration"), trip.get("waitingTime"), trip.get("timeLoss"))
        for trip in root.findall("tripinfo")
    )


@pytest.fixture
def ingolstadt7_kwargs(tmp_path):
    from TREX_comp import states, rewards
    from TREX_comp.config.map_config import map_configs

    map_config = map_configs["ingolstadt7"]
    return dict(
        map_name="ingolstadt7",
        net=os.path.join(str(REPO_ROOT), map_config["net"]),
        state_fn=states.mplight,
        reward_fn=rewards.wait,
        route=None,
        step_length=map_config["step_length"],
        yellow_length=map_config["yellow_length"],
        step_ratio=map_config["step_ratio"],
        end_time=map_config["end_time"],
        max_distance=200,
        lights=map_config["lights"],
        gui=False,
        libsumo=True,
        warmup=map_config["warmup"],
        run=0,
        sumo_seed=4242,
    )


def test_base_scenario_matches_pre_merge_baseenv(ingolstadt7_kwargs, tmp_path):
    old_mod = _load_module("old_base_env_ref", REFERENCE_DIR / "base_env_pre_merge.py")
    from trex_env import TrexEnv

    # Pre-merge BaseEnv builds its output directory as `log_dir + connection_name`
    # (string concatenation, not os.path.join) -- it only works if log_dir already
    # ends in a separator, which is how main.py always calls it in practice
    # (--log_dir defaults to a path ending in os.sep). Matching that here so this
    # test exercises BaseEnv's real, actually-used calling convention rather than
    # an edge case nobody hits.
    old_kwargs = dict(ingolstadt7_kwargs, run_name="regtest-base", log_dir=str(tmp_path / "old") + os.sep, level=None)
    old_name, old_dir = _run_episode(old_kwargs, old_mod.BaseEnv, seed=123, steps=3)

    new_kwargs = dict(ingolstadt7_kwargs, run_name="regtest-base", log_dir=str(tmp_path / "new") + os.sep, incident_config=None)
    from trex_env import TrexEnv as _TrexEnv
    new_name, new_dir = _run_episode(new_kwargs, _TrexEnv, seed=123, steps=3)

    assert old_name == new_name, "connection_name (and therefore output path) diverged"

    # normalize_trailing_comma=True: see _read_metrics' docstring -- pre-merge
    # BaseEnv's CSV writer left a stray trailing ", " per line that TrexEnv
    # intentionally does not reproduce (unified onto IncidentEnv's cleaner
    # format instead). This is the ONLY normalization anywhere in this file;
    # every value/field is still asserted identical.
    old_metrics = _read_metrics(old_dir, old_name, normalize_trailing_comma=True)
    new_metrics = _read_metrics(new_dir, new_name)
    assert old_metrics == new_metrics, "metrics_1.csv differs (beyond the known trailing-comma formatting fix) between pre-merge BaseEnv and TrexEnv(incidents off)"

    old_trips = _read_tripinfo_vehicle_ids_and_durations(old_dir, old_name)
    new_trips = _read_tripinfo_vehicle_ids_and_durations(new_dir, new_name)
    assert old_trips == new_trips, "tripinfo_1.xml (vehicle id/duration/waitingTime/timeLoss) differs"


def test_incident_scenario_matches_pre_merge_incidentenv(ingolstadt7_kwargs, tmp_path):
    old_mod = _load_module("old_incident_env_ref", REFERENCE_DIR / "incident_env_pre_merge.py")
    from trex_env import TrexEnv
    from TREX_comp.config.incident_config import IncidentConfig

    incident_kwargs = dict(ingolstadt7_kwargs, warmup=0)

    old_kwargs = dict(incident_kwargs, run_name="regtest-incident", log_dir=str(tmp_path / "old") + os.sep, level=2)
    old_name, old_dir = _run_episode(old_kwargs, old_mod.IncidentEnv, seed=456, steps=3, pre_seed=(111, 222))

    new_kwargs = dict(
        incident_kwargs, run_name="regtest-incident", log_dir=str(tmp_path / "new") + os.sep,
        incident_config=IncidentConfig(level=2),
    )
    new_name, new_dir = _run_episode(new_kwargs, TrexEnv, seed=456, steps=3, pre_seed=(111, 222))

    assert old_name == new_name, "connection_name (and therefore output path) diverged"

    old_metrics = _read_metrics(old_dir, old_name)
    new_metrics = _read_metrics(new_dir, new_name)
    assert old_metrics == new_metrics, "metrics_1.csv differs between pre-merge IncidentEnv and TrexEnv(incidents on)"

    old_trips = _read_tripinfo_vehicle_ids_and_durations(old_dir, old_name)
    new_trips = _read_tripinfo_vehicle_ids_and_durations(new_dir, new_name)
    assert old_trips == new_trips, "tripinfo_1.xml differs between pre-merge IncidentEnv and TrexEnv(incidents on)"


def test_grid4x4_base_scenario_works_after_route_path_fix(tmp_path):
    """TrexEnv(incident_config=None) must be able to run on a route-based
    network (grid4x4) given the documented (subdirectory) decompression
    layout, unlike pre-merge BaseEnv (see module docstring). This is a
    regression test for the fix itself, not an old-vs-new comparison --
    there is no working "old" run on this network to compare against."""
    grid4x4_dir = REPO_ROOT / "environments" / "grid4x4" / "grid4x4"
    if not grid4x4_dir.is_dir():
        pytest.skip("grid4x4 traffic-flow files not decompressed (see README Installation)")

    from TREX_comp import states, rewards
    from TREX_comp.config.map_config import map_configs
    from trex_env import TrexEnv

    map_config = map_configs["grid4x4"]
    env = TrexEnv(
        run_name="regtest-grid4x4",
        map_name="grid4x4",
        net=os.path.join(str(REPO_ROOT), map_config["net"]),
        state_fn=states.mplight,
        reward_fn=rewards.wait,
        route=os.path.join(str(REPO_ROOT), map_config["route"]),
        step_length=map_config["step_length"],
        yellow_length=map_config["yellow_length"],
        step_ratio=map_config["step_ratio"],
        end_time=map_config["end_time"],
        max_distance=200,
        lights=map_config["lights"],
        gui=False,
        log_dir=str(tmp_path),
        libsumo=True,
        warmup=map_config["warmup"],
        run=0,
        incident_config=None,
        sumo_seed=1,
    )
    obs = env.reset()
    obs, rew, done, info = env.step({ts: 0 for ts in env.signal_ids})
    env.close()
