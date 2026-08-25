# Tests

Unit tests for the scientifically load-bearing pieces of T-REX that don't require a
running SUMO simulation to verify: the AASHTO stopping-sight-distance formula, the ICM
rerouting-probability math, and the bounds/statistics of `Initializer`'s random
incident-parameter sampling (Section 2.3 / Appendix A / Section 2.4.2 of the paper).

`Initializer`/`Deployment` normally require a live SUMO/TraCI connection and a `.net.xml`
file to construct (they call `sumolib.net.readNet(...)` and various `traci.*` functions in
`__init__`). To keep these tests fast and independent of a SUMO install, they build bare
instances via `Class.__new__(Class)`, set only the handful of attributes the method under
test actually reads, and monkeypatch the specific `traci` calls it makes. Run with:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

No SUMO binary is required for this suite. What's **not** covered here: the metric
functions (LSI/FPD/CR/AUC/RAUC/PDI) aren't implemented anywhere in this repository (see
`AUDIT_REPORT.md` Section 0), so there's nothing to unit-test yet; and a true end-to-end
integration smoke test (actually running SUMO via `main.py`) isn't part of this suite --
see `AUDIT_REPORT.md` Section 2.8 for the manual smoke test performed during the audit and
what it found.
