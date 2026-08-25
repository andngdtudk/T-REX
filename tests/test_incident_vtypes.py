"""Regression test for the CAV4 teleport-exemption vType (see AUDIT_REPORT.md).

T_REX.py::Deployment.manage_incident_queue_teleport_exemption switches a queued
vehicle's type to 'CAV4' via traci.vehicle.setType(veh, 'CAV4'). That call fails
at runtime with TraCIException("Vehicle type 'CAV4' is not known") unless the
network's .add.xml file actually defines an *active* (non-commented-out) CAV4
vType with timeToTeleport="-1". This was true only for ingolstadt21 for part of
this audit; every other incident-capable network crashed on the first vehicle
that queued behind a blocked lane. This test parses each network's .add.xml
directly (no SUMO/TraCI needed) so that regression can't silently reappear.

Covers all 8 networks main.py's --map exposes (confirmed against
TREX_comp/config/map_config.py / main.py's argparse choices --
arterial5x5/turin5 also appear in map_config.py but are not reachable via
--map and are out of scope here). cologne1/ingolstadt1 originally had no
.add.xml at all (main.py's --strategy 2 would have failed outright there,
not merely hit the CAV4 issue); both now have one, copied verbatim from
ingolstadt21's (the working original) rather than reintroducing anything
from the unrelated, still-disabled vType-distribution block that predates
CAV4 in every file. See tests/test_cav4_live.py for a live-SUMO test that
each network actually exempts a queued vehicle at runtime, not just that
the vType is declared.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENTS_DIR = REPO_ROOT / "environments"

# Every network reachable via main.py's --map that IncidentEnv/TrexEnv can run
# --strategy 2 on (has, or should have, an .add.xml defining CAV4/IC).
INCIDENT_CAPABLE_NETWORKS = [
    "grid4x4",
    "arterial4x4",
    "cologne1",
    "cologne3",
    "cologne8",
    "ingolstadt1",
    "ingolstadt7",
    "ingolstadt21",
]


def _add_xml_path(network):
    return ENVIRONMENTS_DIR / network / f"{network}.add.xml"


@pytest.mark.parametrize("network", INCIDENT_CAPABLE_NETWORKS)
def test_cav4_teleport_exemption_vtype_is_active(network):
    add_xml_path = _add_xml_path(network)
    assert add_xml_path.exists(), f"{add_xml_path} does not exist"

    root = ET.parse(add_xml_path).getroot()
    # ET only sees *live* elements -- a commented-out <vType id="CAV4" .../>
    # (dead XML, as this was for every network but ingolstadt21 before this
    # fix) is invisible to it, which is exactly the failure mode we're
    # guarding against.
    cav4_types = [vtype for vtype in root.findall("vType") if vtype.get("id") == "CAV4"]

    assert cav4_types, (
        f"{network}: no active <vType id=\"CAV4\"> in {add_xml_path.name} -- "
        "Deployment.manage_incident_queue_teleport_exemption will crash with "
        "TraCIException(\"Vehicle type 'CAV4' is not known\") the first time a "
        "vehicle queues behind a blocked lane on this network."
    )
    assert cav4_types[0].get("timeToTeleport") == "-1", (
        f"{network}: CAV4 vType exists but is missing timeToTeleport=\"-1\", "
        "so it would no longer be exempt from SUMO's global teleport timeout."
    )
