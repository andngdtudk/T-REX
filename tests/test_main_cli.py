"""Unit test for main.py's --strategy 3 fail-fast behavior.

Before this was fixed, --strategy 3 ("curriculum") fell through to
IncidentEnv(..., level=None) and crashed with an opaque
TypeError: 'NoneType' object cannot be interpreted as an integer the first
time _initialize_incidents did range(self.level). main.py now checks for
strategy == 3 explicitly, before any environment is constructed, and raises
a clear NotImplementedError instead. This check happens before any SUMO/env
construction, so this test needs no SUMO installation.
"""
import sys

import pytest

import main as main_module


def test_strategy_3_raises_not_implemented_error_not_type_error(monkeypatch):
    monkeypatch.setenv("LIBSUMO_AS_TRACI", "1")
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--agent", "MAXPRESSURE", "--map", "grid4x4", "--eps", "1",
         "--strategy", "3", "--libsumo", "True"],
    )

    with pytest.raises(NotImplementedError, match=r"--strategy 3"):
        main_module.main()


def test_strategy_1_and_2_are_still_accepted_by_argparse():
    # choices=[1, 2, 3] on --strategy -- confirm 1/2 aren't rejected by argparse
    # itself (only strategy 3 should fail, and only inside main(), not at parse time).
    monkeypatch_argv = ["main.py", "--agent", "MAXPRESSURE", "--map", "grid4x4",
                         "--strategy", "1"]
    orig_argv = sys.argv
    try:
        sys.argv = monkeypatch_argv
        args = main_module.parse_arguments()
        assert args.strategy == 1
    finally:
        sys.argv = orig_argv
