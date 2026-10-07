"""
fish_analyzer/idtrackerai_setup_window.py
=========================================
Open idtracker.ai's own window to make or check a setup - and nothing else.

Run as a script, by path, in a separate process (see tracking.py):

    python idtrackerai_setup_window.py --video V --save-to S.toml [--load L.toml]

It is idtracker.ai's window with two changes, because here the window has one
job and tracking is done afterwards, for the whole folder, from the Tracking
tab:

  - "Close window and track video" becomes "Save setup and close". It writes
    the settings straight to --save-to and tracks nothing.
  - "Save parameters" is hidden, so there is one way to save and no file
    dialog to send the file to the wrong folder.

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


def _customise_window(save_to: Path) -> None:
    import idtrackerai.segmentation_app.main as segmentation
    from idtrackerai.segmentation_app import SegmentationGUI

    toml_format = segmentation.toml_format
    original_init = SegmentationGUI.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def save_and_close():
            parameters = self.out_parameters()
            if self.unacceptable_parameters(parameters):
                return
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

    try:
        _customise_window(args.save_to)
    except Exception as exc:
        print("Could not adapt idtracker.ai's window to this version "
              f"({exc!r}). Opening it unchanged: use \"Save parameters\" and "
              f"save as {args.save_to}. Do not press \"Close window and track "
              "video\".", flush=True)

    sys.argv = ["idtrackerai"]
    if args.load:
        sys.argv += ["--load", args.load]
    sys.argv += ["--video_paths", args.video]

    from idtrackerai.start.__main__ import main as idtrackerai_main
    idtrackerai_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
