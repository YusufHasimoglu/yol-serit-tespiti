"""CLI regressions that do not launch a costly neural-network training run."""
import sys

import egit
import pytest


def test_style_aware_user_training_does_not_require_bdd(monkeypatch, tmp_path):
    """--style-aware is valid with user masks alone; BDD remains optional."""
    monkeypatch.setattr(sys, "argv", ["egit.py", "--style-aware", "--synthetic", "0",
                                       "--epochs", "1", "--batch", "1",
                                       "--real-folder", str(tmp_path / "missing"),
                                       "--out", str(tmp_path / "candidate.pt")])
    # The command reaches its normal minimum-data validation rather than
    # attempting to construct a Path(None) for the absent BDD dataset.
    with pytest.raises(SystemExit) as exc:
        egit.main()
    assert exc.value.code == 2
