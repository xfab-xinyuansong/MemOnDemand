from __future__ import annotations

from memondemand import __version__
from memondemand.cli import main


def test_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_doctor_runs(capsys) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "memondemand" in out
    assert "project_root" in out


def test_top_level_help_runs(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "build-hierarchy" in out
    assert "evaluate" in out
