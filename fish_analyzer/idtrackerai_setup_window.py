"""
fish_analyzer/idtrackerai_setup_window.py
=========================================
Open idtracker.ai's own window to make or check a setup - and nothing else.

Run as a script, by path, in a separate process (see tracking.py):

    python idtrackerai_setup_window.py --video V --save-to S.toml [--load L.toml]

It is idtracker.ai's window with four changes. The first two are because here
the window has one job and tracking is done afterwards, for the whole folder,
from the Tracking tab:

  - "Close window and track video" becomes "Save setup and close". It writes
    the settings straight to --save-to and tracks nothing.
  - "Save parameters" is hidden, so there is one way to save and no file
    dialog to send the file to the wrong folder.
  - Beside the background statistic there is a "Rolling (1 min)" box. Ticked,
    the detection drawn on the video uses a background made from the minute
    around the frame on show (see idtrackerai_rolling.py), for recordings
    whose lighting drifts. It is saved in the setup and tracking does the same.
  - "Next frame with more blobs than animals" moves the video to the next
    frame where the current settings detect too many blobs, which is where a
    threshold needs looking at.

This reaches into idtracker.ai's window class, which is not a public interface.
If a future version renames what it relies on, the window opens unchanged and
says so, rather than failing to open.

Do not import this from the package: it is run by path precisely so that it
does not load fish_analyzer (and with it several seconds of imports).
"""
import argparse
import sys
from pathlib import Path

SAVED_MARKER = "SETUP_SAVED"


def _add_rolling_background(window, start_ticked: bool):
    """Put the "Rolling (1 min)" box beside the background statistic and make
    the detection drawn on the video follow it. Returns the box."""
    import idtrackerai_rolling as rolling
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QCheckBox

    widget, analyzer = window.bkg_widget, window.frame_analyzer
    box = QCheckBox("Rolling\n(1 min)")
    box.setToolTip(
        "For recordings whose lighting drifts, such as sunlight moving across "
        "the tank.\n\nTicked, each stretch of video is compared with a "
        "background made from the minute around it, not one made from the "
        "whole video. Scrub to a late frame to see the difference.\n\n"
        "An animal that stays in one spot for more than half a minute becomes "
        "part of the background and is not detected while it sits there.\n\n"
        "Has no effect unless background subtraction is ticked, or with a "
        "custom background image.")
    box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    box.setChecked(start_ticked)
    widget.layout().insertWidget(widget.layout().indexOf(widget.bkg_stat) + 1, box)
    backgrounds = {}

    def background_for_frame(frame_number: int):
        """The rolling background for the stretch holding this frame - the
        same stretches tracking segments in - or None for a still one."""
        stat = widget.bkg_stat.currentText().lower()
        if not box.isChecked() or stat not in rolling.STATISTICS:
            return None
        episode = next((e for e in getattr(widget, "episodes", [])
                        if e.global_start <= frame_number < e.global_end), None)
        if episode is None:
            return None
        key = (str(episode.video_path), episode.local_start, stat)
        if key not in backgrounds:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                backgrounds[key] = rolling.background_for(
                    episode.video_path, episode.local_start, episode.local_end, stat)
            finally:
                QApplication.restoreOverrideCursor()
        return backgrounds[key]

    def redraw():
        analyzer.need_to_redraw = True
        analyzer.new_parameters.emit()

    analyzer.rolling_background = background_for_frame
    box.stateChanged.connect(lambda _: redraw())
    return box


def _add_jump_to_extra_blobs(window) -> None:
    """Put a button under the blob chart that moves the video to the next
    frame with more blobs than animals, judged with the settings as they are
    now - the same detection the window draws."""
    import cv2
    from idtrackerai.base.animals_detection import process_frame
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMessageBox, QProgressDialog, QPushButton

    button = QPushButton("Next frame with more blobs than animals")
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setToolTip(
        "Looks forward from the frame on show for the next one where these "
        "settings detect more blobs than the number of animals, and goes "
        "there.\nFewer blobs than animals is not looked for: that happens "
        "whenever two animals touch.")
    layout = window.close_and_track_btn.parentWidget().layout()
    layout.insertWidget(layout.indexOf(window.check_segm), button)

    def jump():
        animals = window.n_animals.value()
        episodes = sorted(getattr(window.bkg_widget, "episodes", []),
                          key=lambda e: e.global_start)
        if not animals or not episodes:
            QMessageBox.information(
                window, "Next frame with more blobs than animals",
                "Open a video and set the number of animals first.")
            return
        player, analyzer = window.videoPlayer, window.frame_analyzer
        rolling = getattr(analyzer, "rolling_background", lambda frame: None)
        start = player.current_frame + 1
        progress = QProgressDialog(
            f"Looking for a frame with more than {animals} blobs...", "Stop",
            start, max(start + 1, player.n_frames), window)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(400)
        found, stopped = None, False

        for episode in episodes:
            if episode.global_end <= start:
                continue
            first = max(start, episode.global_start)
            background = analyzer.bkg_model
            if background is not None:
                rolled = rolling(first)
                if rolled is not None and rolled.shape == background.shape:
                    background = rolled
            cap = cv2.VideoCapture(str(episode.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    episode.local_start + first - episode.global_start)
            for frame_number in range(first, episode.global_end):
                ok, frame = cap.read()
                if not ok:
                    break
                if window.blobInfo.in_tracking_intervals(frame_number):
                    areas = process_frame(
                        frame, bkg_model=background, ROI_mask=analyzer.ROI_mask,
                        intensity_ths=analyzer.intensity_ths,
                        area_ths=analyzer.area_ths)[0]
                    if len(areas) > animals:
                        found = frame_number
                        break
                if frame_number % 30 == 0:
                    progress.setValue(frame_number)
                    if progress.wasCanceled():
                        stopped = True
                        break
            cap.release()
            if found is not None or stopped:
                break

        progress.close()
        if found is not None:
            player.setCurrentFrame(found, True)
        elif not stopped:
            QMessageBox.information(
                window, "Next frame with more blobs than animals",
                f"No later frame has more than {animals} blobs with these "
                "settings.")

    button.clicked.connect(jump)


def _draw_with_rolling_background() -> None:
    """Let the part of the window that draws the detection swap in a
    background for the frame it is drawing."""
    from idtrackerai.segmentation_app.widgets.frame_analyzer import FrameAnalyzer

    original_paint = FrameAnalyzer.paint_on_canvas
    original_process = FrameAnalyzer.process_frame

    def paint_on_canvas(self, painter, frame_number, frame):
        self.frame_being_drawn = frame_number
        return original_paint(self, painter, frame_number, frame)

    def process_frame(self, frame):
        still = self.bkg_model
        source = getattr(self, "rolling_background", None)
        if frame is not None and still is not None and source is not None:
            rolled = source(getattr(self, "frame_being_drawn", 0))
            if rolled is not None and rolled.shape == still.shape:
                self.bkg_model = rolled
        try:
            return original_process(self, frame)
        finally:
            self.bkg_model = still

    FrameAnalyzer.paint_on_canvas = paint_on_canvas
    FrameAnalyzer.process_frame = process_frame


def _customise_window(save_to: Path, rolling_ticked: bool) -> None:
    import idtrackerai.segmentation_app.main as segmentation
    import idtrackerai_rolling as rolling
    from idtrackerai.segmentation_app import SegmentationGUI

    toml_format = segmentation.toml_format
    original_init = SegmentationGUI.__init__
    _draw_with_rolling_background()

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        rolling_box = _add_rolling_background(self, rolling_ticked)
        _add_jump_to_extra_blobs(self)

        def save_and_close():
            parameters = self.out_parameters()
            if self.unacceptable_parameters(parameters):
                return
            stat = parameters.get("background_subtraction_stat")
            if rolling_box.isChecked() and stat in rolling.STATISTICS:
                parameters["background_subtraction_stat"] = rolling.PREFIX + stat
            # The setup is used for every video in the folder; the Tracking
            # tab supplies the video and the session name each time.
            parameters.pop("video_paths", None)
            parameters.pop("name", None)
            try:
                with open(save_to, "w", encoding="utf_8") as file:
                    for key, value in parameters.items():
                        file.write(f"{key} = {toml_format(value)}\n")
            except OSError as exc:
                # An exception escaping a Qt slot ends the process without a
                # word, which would look like the window simply vanishing.
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self, "The setup could not be saved",
                    f"Could not write {save_to}\n\n{exc}\n\n"
                    "Nothing was saved. Check the folder is not read-only, "
                    "then press the button again.")
                return
            print(f"{SAVED_MARKER} {save_to}", flush=True)
            self.close()

        button = self.close_and_track_btn
        button.clicked.disconnect()
        button.clicked.connect(save_and_close)
        button.setText("Save setup and close")
        button.setToolTip(
            f"Save these settings to {save_to.name} and close this window.\n"
            "Tracking is started afterwards from the Free Swim Analyzer.")
        self.save_parameters.hide()

    SegmentationGUI.__init__ = init


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--save-to", required=True, type=Path)
    parser.add_argument("--load")
    args = parser.parse_args()

    # idtracker.ai would take a rolling statistic for a file name, so it is
    # given the plain one and the box is ticked instead.
    rolling_ticked, stat = False, ""
    if args.load:
        import idtrackerai_rolling
        rolling_ticked, stat = idtrackerai_rolling.setup_statistic(Path(args.load))

    try:
        _customise_window(args.save_to, rolling_ticked)
    except Exception as exc:
        print("Could not adapt idtracker.ai's window to this version "
              f"({exc!r}). Opening it unchanged: use \"Save parameters\" and "
              f"save as {args.save_to}. Do not press \"Close window and track "
              "video\".", flush=True)

    sys.argv = ["idtrackerai"]
    if args.load:
        sys.argv += ["--load", args.load]
    if rolling_ticked:
        sys.argv += ["--background_subtraction_stat", stat]
    sys.argv += ["--video_paths", args.video]

    from idtrackerai.start.__main__ import main as idtrackerai_main
    idtrackerai_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
