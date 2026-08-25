# Contributing to T-REX

Thanks for your interest in contributing! T-REX is the reference implementation
accompanying a published research paper (see `README.md` for the citation), so
contributions that touch the scientific core — the `Initializer`/`Deployment`
incident model in `T_REX.py` (ICM rerouting, SSD speed adaptation, incident
sampling), the metric definitions, or any RL agent's MDP definition — should be
held to a higher bar than ordinary bug fixes: please open an issue describing
the change and its motivation *before* submitting a large pull request, so it
can be discussed against the paper's methodology first.

## Getting started

1. Fork the repository and create a feature branch off `main`.
2. Follow the installation steps in `README.md` (SUMO + `pip install -r requirements-dev.txt`).
3. Make your changes.
4. Run the test suite: `pytest tests/`.
5. If you changed formatting-sensitive code, you can optionally run
   `ruff check .`, `black .`, and `isort .` — these are advisory in CI for now
   (see `AUDIT_REPORT.md`), not required, but appreciated.
6. Open a pull request describing what changed and why.

## Reporting bugs

Please include: the command you ran, the network/agent you used, the full
traceback, and your SUMO/Python/package versions (`pip freeze`). If the bug
affects reproducing a specific number from the paper, please say which
experiment/table it corresponds to.

## Code style

- Prefer `logging` over `print()` for anything beyond a one-off local debugging session.
- New scientifically load-bearing logic (metrics, sampling distributions, reward/MDP
  definitions, ICM/SSD formulas) should include a docstring citing the relevant paper
  section/equation, and ideally a unit test under `tests/` (see `tests/README.md` for the
  pattern used to test `Initializer`/`Deployment` methods without a live SUMO connection).
- Don't hardcode a value that's already defined in a config module
  (`TREX_comp/config/*.py`) — import it instead.

## Adding a new RL-TSC method or network

See the "Extending T-REX" section of `README.md`.
