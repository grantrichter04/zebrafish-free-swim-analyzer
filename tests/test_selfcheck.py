"""The setup self-check: what install.bat and the Check Setup button report.

Nothing here needs idtracker.ai, torch or a GPU - the lookups are replaced -
so it runs the same in CI as on the lab laptop.
"""
import fish_analyzer
from fish_analyzer import selfcheck


def _with(monkeypatch, versions, gpu):
    monkeypatch.setattr(selfcheck, "_package_version", versions.get)
    monkeypatch.setattr(selfcheck, "_gpu_status", lambda: gpu)
    return {item.name: item for item in selfcheck.run_checks()}


FULL = {"opencv-python-headless": "5.0.0.93", "idtrackerai": "6.0.14",
        "torch": "2.11.0+cu128"}


def test_fully_set_up_laptop_passes_everything(monkeypatch):
    items = _with(monkeypatch, FULL, (True, "NVIDIA GeForce RTX 4070"))

    assert [i.name for i in selfcheck.run_checks()] == [
        "Python", "Analyzer", "OpenCV", "idtracker.ai", "PyTorch", "GPU"]
    assert all(i.ok for i in items.values())
    assert items["Analyzer"].detail == fish_analyzer.__version__
    assert items["GPU"].detail == "NVIDIA GeForce RTX 4070"
    assert selfcheck.exit_code(list(items.values())) == 0


def test_missing_idtrackerai_is_a_warning_not_a_failure(monkeypatch):
    """Analysis still works without tracking, so this must not fail the check."""
    items = _with(monkeypatch, {"opencv-python-headless": "5.0.0.93"},
                  (False, "PyTorch is not installed"))

    assert not items["idtracker.ai"].ok
    assert not items["idtracker.ai"].required
    assert "not installed" in items["idtracker.ai"].detail
    assert not items["GPU"].ok
    assert selfcheck.exit_code(list(items.values())) == 0


def test_old_python_fails_the_check(monkeypatch):
    monkeypatch.setattr(selfcheck, "_python_version", lambda: (3, 9, 18))
    items = _with(monkeypatch, FULL, (True, "GPU"))

    assert not items["Python"].ok
    assert items["Python"].required
    assert selfcheck.exit_code(list(items.values())) == 1


def test_report_says_what_is_wrong_in_plain_words(monkeypatch):
    items = _with(monkeypatch, {"opencv-python-headless": "5.0.0.93"},
                  (False, "PyTorch is not installed"))
    report = selfcheck.format_report(list(items.values()))

    assert "[ OK ] Analyzer" in report
    assert "[WARN] idtracker.ai" in report
    assert "Analysis will work. Tracking will not" in report


def test_report_for_a_ready_laptop(monkeypatch):
    items = _with(monkeypatch, FULL, (True, "GPU"))
    assert "Everything is ready." in selfcheck.format_report(list(items.values()))


def test_regular_opencv_counts_when_headless_is_absent(monkeypatch):
    items = _with(monkeypatch, {"opencv-python": "4.12.0"}, (False, "x"))
    assert items["OpenCV"].ok and items["OpenCV"].detail == "4.12.0"


def test_check_flag_prints_the_report_and_does_not_open_the_gui(
        monkeypatch, capsys):
    from fish_analyzer import __main__ as entry

    monkeypatch.setattr(selfcheck, "_package_version", FULL.get)
    monkeypatch.setattr(selfcheck, "_gpu_status", lambda: (True, "GPU"))
    monkeypatch.setattr(fish_analyzer, "EnhancedFishAnalyzer",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("--check must not start the GUI")))

    assert entry.main(["--check"]) == 0
    assert "Everything is ready." in capsys.readouterr().out


def test_check_flag_returns_nonzero_when_a_required_check_fails(monkeypatch):
    from fish_analyzer import __main__ as entry

    monkeypatch.setattr(selfcheck, "_python_version", lambda: (3, 9, 18))
    monkeypatch.setattr(selfcheck, "_package_version", FULL.get)
    monkeypatch.setattr(selfcheck, "_gpu_status", lambda: (True, "GPU"))

    assert entry.main(["--check"]) == 1
