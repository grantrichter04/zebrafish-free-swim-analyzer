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
    app._tracking_live.clear()
    app.tracking_progress_var.set("")
    app._tracking_refresh()


def _rows(app):
    tree = app.tracking_videos_tree
    return [tuple(tree.item(i, "values"))[:2] for i in tree.get_children()]


def _tracked_with(app):
    tree = app.tracking_videos_tree
    return {tree.set(i, "video"): tree.set(i, "setup") for i in tree.get_children()}


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


# --- tracking the folder ------------------------------------------------------

def _fake_tracking(monkeypatch, fail=(), on_start=None):
    """Stand in for idtracker.ai tracking: make the session folder, or fail."""
    tracked = []

    def run(video, setup, on_line, should_stop=lambda: False,
            should_finish_training=lambda: False):
        tracked.append((video.name, setup.name))
        if on_start:
            on_start(video)
        on_line(f"tracking {video.name}")
        stopped = should_stop()
        ok = video.name not in fail and not stopped
        if ok:
            target = tracking.session_folder_for(video) / "trajectories"
            target.mkdir(parents=True)
            (target / "trajectories.npy").write_bytes(b"")
        else:
            tracking.session_folder_for(video).mkdir(exist_ok=True)
        return tracking.TrackOutcome(video, ok, stopped, 1.0,
                                     ["Starting", "Too many blobs   run.py:9"])

    monkeypatch.setattr(tracking, "run_tracking", run)
    return tracked


@pytest.fixture
def reports(tab, monkeypatch):
    seen = []
    monkeypatch.setattr(
        tab, "_report_batch_outcome",
        lambda what, total, ok, failed, degraded: seen.append((total, ok, failed)))
    return seen


def test_track_all_tracks_only_the_untracked_videos(
        tab, experiment, dialogs, reports, monkeypatch):
    (experiment / "third.avi").write_bytes(b"")
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    tracked = _fake_tracking(monkeypatch)
    tab._tracking_set_folder(experiment)
    assert tab.tracking_track_button["state"] == "normal"
    assert tab.tracking_stop_button["state"] == "disabled"

    tab._tracking_track_all()
    _wait_until_idle(tab)

    assert tracked == [("exp.avi", "rig.toml"), ("third.avi", "rig.toml")]
    assert [status for _, status in _rows(tab)] == ["tracked"] * 3
    assert reports == [(2, ["exp.avi", "third.avi"], [])]
    assert "2 of 2 video(s) tracked" in tab.tracking_progress_var.get()
    assert tab.tracking_track_button["state"] == "disabled", "nothing left to track"
    assert "tracking third.avi" in tab.tracking_log_text.get("1.0", "end")


def test_one_failed_video_does_not_stop_the_rest(
        tab, experiment, dialogs, reports, monkeypatch):
    (experiment / "third.avi").write_bytes(b"")
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    tracked = _fake_tracking(monkeypatch, fail={"exp.avi"})
    tab._tracking_set_folder(experiment)

    tab._tracking_track_all()
    _wait_until_idle(tab)

    assert [name for name, _ in tracked] == ["exp.avi", "third.avi"]
    assert dict(_rows(tab)) == {"control.avi": "tracked", "exp.avi": "failed",
                                "third.avi": "tracked"}
    assert reports == [(2, ["third.avi"], ["exp.avi: Too many blobs"])]
    assert tab.tracking_track_button["state"] == "normal", "exp.avi can be retried"


def test_stop_ends_the_batch_after_the_current_video(
        tab, experiment, dialogs, reports, monkeypatch):
    (experiment / "third.avi").write_bytes(b"")
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *a, **k: True)

    def press_stop(video):
        tab._tracking_batch_running = True
        tab._tracking_stop = True

    tracked = _fake_tracking(monkeypatch, on_start=press_stop)
    tab._tracking_set_folder(experiment)

    tab._tracking_track_all()
    _wait_until_idle(tab)

    assert [name for name, _ in tracked] == ["exp.avi"], "third.avi never started"
    assert dict(_rows(tab))["exp.avi"] == "stopped"
    assert dict(_rows(tab))["third.avi"] == "not tracked"
    assert reports == [], "a deliberate stop is not reported as a failure"
    assert "Stopped before the rest" in tab.tracking_progress_var.get()
    assert not tab._tracking_stop


def test_tracking_needs_a_setup(tab, experiment):
    tab._tracking_set_folder(experiment)
    assert tab.tracking_track_button["state"] == "disabled"
    assert _tracked_with(tab)["exp.avi"] == "no setup yet"


# --- a video with a setup of its own ------------------------------------------

def test_a_video_can_be_given_its_own_setup(tab, experiment, dialogs, monkeypatch):
    """One tank has five fish, or was moved: it starts from the shared setup,
    is saved under the video's name, and the others are left alone."""
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    calls = _fake_idtrackerai(monkeypatch)
    tab._tracking_set_folder(experiment)
    tab.tracking_videos_tree.selection_set("1")

    tab._tracking_configure_own()
    _wait_until_idle(tab)

    command = calls[0]
    assert command[command.index("--video") + 1] == str(experiment / "exp.avi")
    assert command[command.index("--load") + 1] == str(experiment / "rig.toml")
    assert command[command.index("--save-to") + 1] == str(experiment / "exp.toml")
    assert (experiment / "rig.toml").read_text() == "number_of_animals = 6\n"
    assert _tracked_with(tab) == {"control.avi": "rig.toml",
                                  "exp.avi": "exp.toml (its own)"}
    assert list(tab.tracking_setup_combo["values"]) == ["rig.toml"]
    assert "its own setup" in tab.tracking_hint_var.get()
    assert dialogs["asked"] == [], "an own setup is not offered to other videos"


def test_checking_a_video_opens_the_setup_it_is_tracked_with(
        tab, experiment, dialogs, monkeypatch):
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    (experiment / "exp.toml").write_text("number_of_animals = 5\n")
    calls = _fake_idtrackerai(monkeypatch, save=False)
    tab._tracking_set_folder(experiment)
    tab.tracking_videos_tree.selection_set("1")

    tab._tracking_configure(edit=True)
    _wait_until_idle(tab)

    command = calls[0]
    assert command[command.index("--load") + 1] == str(experiment / "exp.toml")
    assert command[command.index("--save-to") + 1] == str(experiment / "exp.toml")


def test_each_video_is_tracked_with_its_own_setup_if_it_has_one(
        tab, experiment, dialogs, reports, monkeypatch):
    (experiment / "third.avi").write_bytes(b"")
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    (experiment / "exp.toml").write_text("number_of_animals = 5\n")
    tracked = _fake_tracking(monkeypatch)
    tab._tracking_set_folder(experiment)

    tab._tracking_track_all()
    _wait_until_idle(tab)

    assert tracked == [("exp.avi", "exp.toml"), ("third.avi", "rig.toml")]


# --- finishing identity training early ----------------------------------------

def test_identity_training_can_be_finished_from_the_tab(
        tab, experiment, dialogs, reports, monkeypatch):
    """The button is live only while idtracker.ai's training table is
    scrolling, and pressing it reaches the run of the video in progress."""
    (experiment / "rig.toml").write_text("number_of_animals = 6\n")
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *a, **k: True)
    seen = {}

    def run(video, setup, on_line, should_stop=lambda: False,
            should_finish_training=lambda: False):
        on_line("12:53:15 Batch | Batches/s | Silhouette score | Too far apart")
        on_line("12:55:02   600 |   12.3    |      0.8312!     |      4.1%  |  9.8%")
        deadline = time.monotonic() + 10
        while not should_finish_training() and time.monotonic() < deadline:
            time.sleep(0.02)
        seen["asked"] = should_finish_training()
        on_line("         Training interrupted by user")
        target = tracking.session_folder_for(video) / "trajectories"
        target.mkdir(parents=True)
        (target / "trajectories.npy").write_bytes(b"")
        return tracking.TrackOutcome(video, True, False, 1.0, [])

    monkeypatch.setattr(tracking, "run_tracking", run)
    tab._tracking_set_folder(experiment)
    assert tab.tracking_finish_button["state"] == "disabled"

    tab._tracking_track_all()
    deadline = time.monotonic() + 10
    while tab._tracking_silhouette is None and time.monotonic() < deadline:
        tab.root.update()
        time.sleep(0.02)
    assert tab.tracking_finish_button["state"] == "normal"
    tab._tracking_show_progress()
    assert "silhouette score 0.8312" in tab.tracking_progress_var.get()

    tab._tracking_request_finish_training()
    assert tab.tracking_finish_button["state"] == "disabled", "asked once"
    _wait_until_idle(tab)

    assert seen["asked"]
    assert tab.tracking_finish_button["state"] == "disabled"
    assert not tab._tracking_finish_training, "does not carry to the next batch"


def test_tracked_sessions_load_into_the_analysis_tabs(
        tab, experiment, synthetic_npy, monkeypatch):
    import shutil
    shutil.copy(synthetic_npy, experiment / "session_control" / "trajectories"
                / "trajectories.npy")
    shown = []
    monkeypatch.setattr("tkinter.messagebox.showinfo",
                        lambda title, text: shown.append(text))
    tab._tracking_set_folder(experiment)
    assert tab.tracking_load_button["state"] == "normal"

    tab._tracking_load_sessions()

    assert list(tab.loaded_files) == ["control"]
    assert tab.loaded_files["control"].n_fish == 3
    assert tab.loaded_files["control"].calibration.unit_name == "BL"
    assert "Loaded 1 session(s)" in shown[0]
    assert tab.notebook.select() == str(tab.data_tab_frame)

    tab._tracking_load_sessions()
    assert "Loaded 0 session(s)" in shown[1] and "control" in shown[1]


# --- reviewing in the validator ---------------------------------------------------

def test_corrections_saved_in_the_validator_reload_a_loaded_session(
        tab, experiment, synthetic_npy, synthetic_npy_larger_fish, dialogs, monkeypatch):
    import json
    import shutil
    session = experiment / "session_control"
    trajectories = session / "trajectories" / "trajectories.npy"
    shutil.copy(synthetic_npy, trajectories)
    (session / "session.json").write_text("{}")
    told = []
    monkeypatch.setattr("tkinter.messagebox.showinfo",
                        lambda title, text: told.append(title))
    tab._tracking_set_folder(experiment)
    tab._tracking_load_sessions()
    assert tab.loaded_files["control"].metadata.body_length == 40.0
    tab.loaded_files["control"].processed_data = ["old results"]

    def validator(command, cwd, on_line, should_stop=lambda: False):
        assert command[-1] == str(session)
        shutil.copy(synthetic_npy_larger_fish, trajectories)   # "Ctrl+S"
        (session / "session.json").write_text(
            json.dumps({"last_validated": "2026-10-08T11:00:00"}))
        return 0

    monkeypatch.setattr(tracking, "run_process", validator)
    tab.sessions_tree.selection_set("control")
    assert tab.sessions_tree.set("control", "reviewed") == ""

    tab._review_selected_session()
    _wait_until_idle(tab)

    assert tab.loaded_files["control"].metadata.body_length == 52.0, "reloaded"
    assert tab.loaded_files["control"].processed_data is None
    assert "Session reloaded" in told
    assert dict(_rows(tab))["control.avi"] == "tracked, reviewed 2026-10-08"
    assert tab.sessions_tree.set("control", "reviewed") == "\u2713 2026-10-08"
    assert "corrections saved" in tab.tracking_hint_var.get()


def test_a_review_that_changes_nothing_leaves_loaded_sessions_alone(
        tab, experiment, synthetic_npy, dialogs, monkeypatch):
    import shutil
    shutil.copy(synthetic_npy, experiment / "session_control" / "trajectories"
                / "trajectories.npy")
    (experiment / "session_control" / "session.json").write_text("{}")
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *a, **k: None)
    tab._tracking_set_folder(experiment)
    tab._tracking_load_sessions()
    tab.loaded_files["control"].processed_data = []
    monkeypatch.setattr(tracking, "run_process", lambda *a, **k: 0)
    tab.sessions_tree.selection_set("control")

    tab._review_selected_session()
    _wait_until_idle(tab)

    assert tab.loaded_files["control"].processed_data == []
    assert "nothing was changed" in tab.tracking_hint_var.get()


def test_a_loaded_session_can_be_reviewed_from_the_sessions_table(
        tab, experiment, synthetic_npy, dialogs, monkeypatch):
    """The sessions table is where tracking quality is shown, so the review
    is reachable from there too - including for a renamed session."""
    import shutil
    session = experiment / "session_control"
    shutil.copy(synthetic_npy, session / "trajectories" / "trajectories.npy")
    (session / "session.json").write_text("{}")
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *a, **k: None)
    tab._tracking_set_folder(experiment)
    tab._tracking_load_sessions()
    tab._rename("control", "ctrl")
    opened = []
    monkeypatch.setattr(tracking, "run_process",
                        lambda command, *a, **k: opened.append(command[-1]) or 0)
    tab.sessions_tree.selection_set("ctrl")

    tab._review_selected_session()
    _wait_until_idle(tab)

    assert opened == [str(session)]


def test_review_explains_itself_when_the_folder_is_not_a_full_session(
        tab, experiment, synthetic_npy, monkeypatch):
    told = []
    monkeypatch.setattr("tkinter.messagebox.showinfo",
                        lambda title, text: told.append(text))
    monkeypatch.setattr(tracking, "run_process",
                        lambda *a, **k: pytest.fail("the validator was started"))

    tab._review_session(experiment / "session_control", "control")

    assert "not a complete idtracker.ai session" in told[0]
