"""The Tracking tab: what it shows for a folder, and configuring a setup.

idtracker.ai is never started. tracking.run_process is replaced by a stand-in
that does what a user would do in idtracker.ai's window: save a .toml, or not.
"""
import time

import pytest

from fish_analyzer import tracking


@pytest.fixture
def experiment(tmp_path):
    (tmp_path / "control.avi").write_bytes(b"")
    (tmp_path / "exp.avi").write_bytes(b"")
    done = tmp_path / "session_control" / "trajectories"
    done.mkdir(parents=True)
    (done / "trajectories.npy").write_bytes(b"")
    return tmp_path


@pytest.fixture
def tab(app, monkeypatch):
    monkeypatch.setattr(tracking, "idtrackerai_available", lambda: True)
    yield app
    app._tracking_folder = None
    app._tracking_refresh()


def _rows(app):
    tree = app.tracking_videos_tree
    return [tuple(tree.item(i, "values")) for i in tree.get_children()]


def _wait_until_idle(app, timeout=10):
    deadline = time.monotonic() + timeout
    while app._tracking_busy and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.02)
    assert not app._tracking_busy, "the tracking tab never finished"


def test_choosing_a_folder_lists_videos_with_their_status(tab, experiment):
    tab._tracking_set_folder(experiment)

    assert _rows(tab) == [("control.avi", "tracked"), ("exp.avi", "not tracked")]
    assert tab.tracking_setup_combo["values"] in ("", ())
    assert "no setup yet" in tab.tracking_hint_var.get()
    assert tab.tracking_configure_button["state"] == "normal"
    assert tab.tracking_edit_button["state"] == "disabled"


def test_existing_setups_are_offered(tab, experiment):
    (experiment / "ir_rig.toml").write_text("number_of_animals = 8\n")
    tab._tracking_set_folder(experiment)

    assert list(tab.tracking_setup_combo["values"]) == ["ir_rig.toml"]
    assert tab.tracking_setup_var.get() == "ir_rig.toml"
    assert tab.tracking_edit_button["state"] == "normal"


def test_buttons_are_disabled_without_idtrackerai(app, experiment, monkeypatch):
    monkeypatch.setattr(tracking, "idtrackerai_available", lambda: False)
    app._tracking_set_folder(experiment)
    try:
        assert app.tracking_configure_button["state"] == "disabled"
    finally:
        app._tracking_folder = None
        app._tracking_refresh()


def test_configuring_selects_the_setup_the_user_saved(tab, experiment, monkeypatch):
    seen = {}

    def fake_idtrackerai(command, cwd, on_line, should_stop=lambda: False):
        seen["command"], seen["cwd"] = command, cwd
        on_line("Welcome to idtracker.ai")
        (experiment / "control.toml").write_text("number_of_animals = 8\n")
        return 0

    monkeypatch.setattr(tracking, "run_process", fake_idtrackerai)
    monkeypatch.setattr("tkinter.messagebox.askokcancel", lambda *a, **k: True)
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)
    assert tab.tracking_configure_button["state"] == "disabled"
    _wait_until_idle(tab)

    assert seen["cwd"] == experiment, "idtracker.ai suggests saving in its cwd"
    assert str(experiment / "control.avi") in seen["command"]
    assert "--track" not in seen["command"]
    assert tab.tracking_setup_var.get() == "control.toml"
    assert "Setup saved: control.toml" in tab.tracking_hint_var.get()
    assert "Welcome to idtracker.ai" in tab.tracking_log_text.get("1.0", "end")
    assert tab.tracking_configure_button["state"] == "normal"


def test_closing_idtrackerai_without_saving_says_so(tab, experiment, monkeypatch):
    warnings = []
    monkeypatch.setattr(tracking, "run_process", lambda *a, **k: 0)
    monkeypatch.setattr("tkinter.messagebox.askokcancel", lambda *a, **k: True)
    monkeypatch.setattr("tkinter.messagebox.showwarning",
                        lambda title, text: warnings.append(title))
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)
    _wait_until_idle(tab)

    assert warnings == ["No setup was saved"]
    assert tab.tracking_setup_var.get() == ""


def test_cancelling_the_instructions_starts_nothing(tab, experiment, monkeypatch):
    monkeypatch.setattr(tracking, "run_process",
                        lambda *a, **k: pytest.fail("idtracker.ai was started"))
    monkeypatch.setattr("tkinter.messagebox.askokcancel", lambda *a, **k: False)
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)

    assert not tab._tracking_busy
