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
    app._tracking_checked.clear()
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


@pytest.fixture
def dialogs(monkeypatch):
    """Answer the tab's dialogs: name the setup 'rig', accept the
    instructions, and decline (by default) to check the next video."""
    answers = {"name": "rig", "check_next": [], "warnings": [], "asked": []}
    monkeypatch.setattr("tkinter.simpledialog.askstring",
                        lambda *a, **k: answers["name"])
    monkeypatch.setattr("tkinter.messagebox.askokcancel", lambda *a, **k: True)
    monkeypatch.setattr("tkinter.messagebox.showwarning",
                        lambda title, text: answers["warnings"].append(title))

    def askyesno(title, text):
        answers["asked"].append(text)
        return answers["check_next"].pop(0) if answers["check_next"] else False
    monkeypatch.setattr("tkinter.messagebox.askyesno", askyesno)
    return answers


def _fake_idtrackerai(monkeypatch, save=True):
    """Stand in for idtracker.ai's window: record the command, and press
    'Save setup and close' (or not)."""
    calls = []

    def run(command, cwd, on_line, should_stop=lambda: False):
        calls.append(command)
        on_line("Welcome to idtracker.ai")
        if save:
            target = command[command.index("--save-to") + 1]
            with open(target, "a") as file:
                file.write("number_of_animals = 6\n")
        return 0

    monkeypatch.setattr(tracking, "run_process", run)
    return calls


def test_configuring_saves_the_named_setup_and_selects_it(
        tab, experiment, dialogs, monkeypatch):
    calls = _fake_idtrackerai(monkeypatch)
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)
    assert tab.tracking_configure_button["state"] == "disabled"
    _wait_until_idle(tab)

    command = calls[0]
    assert command[command.index("--video") + 1] == str(experiment / "control.avi")
    assert command[command.index("--save-to") + 1] == str(experiment / "rig.toml")
    assert "--load" not in command and "--track" not in command
    assert tab.tracking_setup_var.get() == "rig.toml"
    assert "Setup saved: rig.toml" in tab.tracking_hint_var.get()
    assert "Welcome to idtracker.ai" in tab.tracking_log_text.get("1.0", "end")
    assert tab.tracking_configure_button["state"] == "normal"


def test_after_saving_the_next_video_is_offered_for_a_check(
        tab, experiment, dialogs, monkeypatch):
    calls = _fake_idtrackerai(monkeypatch, save=True)
    dialogs["check_next"] = [True]
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)
    _wait_until_idle(tab)
    _wait_until_idle(tab)

    assert len(calls) == 2, "the setup should have been opened on the 2nd video"
    assert "exp.avi" in dialogs["asked"][0]
    second = calls[1]
    assert second[second.index("--video") + 1] == str(experiment / "exp.avi")
    assert second[second.index("--load") + 1] == str(experiment / "rig.toml")
    assert second[second.index("--save-to") + 1] == str(experiment / "rig.toml")
    assert "Checked on all 2 video(s)" in tab.tracking_hint_var.get()
    assert len(dialogs["asked"]) == 1, "no video left to offer"


def test_checking_without_changes_leaves_the_setup_alone(
        tab, experiment, dialogs, monkeypatch):
    setup = experiment / "rig.toml"
    setup.write_text("number_of_animals = 6\n")
    calls = _fake_idtrackerai(monkeypatch, save=False)
    tab._tracking_set_folder(experiment)
    tab.tracking_videos_tree.selection_set("1")

    tab._tracking_configure(edit=True)
    _wait_until_idle(tab)

    assert calls[0][calls[0].index("--video") + 1] == str(experiment / "exp.avi")
    assert setup.read_text() == "number_of_animals = 6\n"
    assert dialogs["warnings"] == []
    assert "Setup unchanged: rig.toml" in tab.tracking_hint_var.get()


def test_closing_idtrackerai_without_saving_says_so(
        tab, experiment, dialogs, monkeypatch):
    _fake_idtrackerai(monkeypatch, save=False)
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)
    _wait_until_idle(tab)

    assert dialogs["warnings"] == ["No setup was saved"]
    assert tab.tracking_setup_var.get() == ""
    assert dialogs["asked"] == []


def test_a_new_setup_cannot_take_an_existing_name(
        tab, experiment, dialogs, monkeypatch):
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    names = iter(["rig", "bad/name", None])
    monkeypatch.setattr("tkinter.simpledialog.askstring",
                        lambda *a, **k: next(names))
    monkeypatch.setattr(tracking, "run_process",
                        lambda *a, **k: pytest.fail("idtracker.ai was started"))
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)

    assert dialogs["warnings"] == ["That name is taken", "That name cannot be used"]
    assert not tab._tracking_busy


def test_cancelling_the_instructions_starts_nothing(
        tab, experiment, dialogs, monkeypatch):
    monkeypatch.setattr(tracking, "run_process",
                        lambda *a, **k: pytest.fail("idtracker.ai was started"))
    monkeypatch.setattr("tkinter.messagebox.askokcancel", lambda *a, **k: False)
    tab._tracking_set_folder(experiment)

    tab._tracking_configure(edit=False)

    assert not tab._tracking_busy
