"""Finding again the outlines idtracker.ai segmented, from its session folder."""
import json

import cv2
import numpy as np

from fish_analyzer.fish_outlines import OutlineFinder


def _session(folder, settings, background=None, roi=None):
    (folder / "preprocessing").mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps(settings))
    if background is not None:
        cv2.imwrite(str(folder / "preprocessing" / "background.png"), background)
    if roi is not None:
        cv2.imwrite(str(folder / "preprocessing" / "ROI_mask.png"), roi)
    return folder


def _frame_with_fish():
    """A pale tank with a 30 x 10 dark fish, a speck, and a second fish."""
    frame = np.full((200, 300), 200, dtype=np.uint8)
    frame[50:60, 40:70] = 80          # fish, 300 px
    frame[100:102, 100:102] = 80      # speck, too small to be a fish
    frame[150:160, 220:250] = 80      # fish near the right edge
    return frame


def test_background_subtraction_finds_what_differs_by_more_than_the_threshold(tmp_path):
    background = np.full((200, 300), 200, dtype=np.uint8)
    folder = _session(tmp_path / "s", {"intensity_ths": [43, 255],
                                       "area_ths": [50.0, float("inf")],
                                       "use_bkg": True}, background)
    finder = OutlineFinder.for_session(folder)

    outlines = finder.find(_frame_with_fish())

    assert len(outlines) == 2, "two fish; the speck is under the minimum area"
    boxes = sorted(cv2.boundingRect(c) for c in outlines)
    assert boxes[0] == (40, 50, 30, 10)

    faint = np.full((200, 300), 200, dtype=np.uint8)
    faint[50:60, 40:70] = 170         # 30 grey levels: under the threshold of 43
    assert finder.find(faint) == []


def test_the_region_of_interest_hides_what_is_outside_it(tmp_path):
    background = np.full((200, 300), 200, dtype=np.uint8)
    roi = np.zeros((200, 300), dtype=np.uint8)
    roi[:, :150] = 255
    folder = _session(tmp_path / "s", {"intensity_ths": [43, 255],
                                       "area_ths": [50.0, 1e9], "use_bkg": True},
                      background, roi)

    outlines = OutlineFinder.for_session(folder).find(_frame_with_fish())

    assert [cv2.boundingRect(c)[0] for c in outlines] == [40]


def test_without_a_background_the_thresholds_are_a_brightness_range(tmp_path):
    folder = _session(tmp_path / "s", {"intensity_ths": [0, 120],
                                       "area_ths": [50.0, 1e9], "use_bkg": False})
    assert len(OutlineFinder.for_session(folder).find(_frame_with_fish())) == 2


def test_a_colour_frame_gives_the_same_outlines_as_a_grey_one(tmp_path):
    background = np.full((200, 300), 200, dtype=np.uint8)
    folder = _session(tmp_path / "s", {"intensity_ths": [43, 255],
                                       "area_ths": [50.0, 1e9], "use_bkg": True},
                      background)
    finder = OutlineFinder.for_session(folder)
    grey = _frame_with_fish()
    colour = np.stack([grey, grey, grey], axis=-1)
    assert len(finder.find(colour)) == len(finder.find(grey)) == 2


def test_a_session_tracked_at_reduced_resolution_still_lines_up(tmp_path):
    """The saved background is the reduced size; the outlines must come back
    in the pixels of the full frame they are drawn on."""
    background = np.full((100, 150), 200, dtype=np.uint8)
    folder = _session(tmp_path / "s", {"intensity_ths": [43, 255],
                                       "area_ths": [10.0, 1e9], "use_bkg": True},
                      background)

    outlines = OutlineFinder.for_session(folder).find(_frame_with_fish())

    x, y, w, h = sorted(cv2.boundingRect(c) for c in outlines)[0]
    assert abs(x - 40) <= 2 and abs(y - 50) <= 2 and abs(w - 30) <= 3 and abs(h - 10) <= 3


def test_no_finder_when_the_session_does_not_say_how_it_was_segmented(tmp_path):
    assert OutlineFinder.for_session(tmp_path / "nothing here") is None
    no_background = _session(tmp_path / "s", {"intensity_ths": [43, 255],
                                              "area_ths": [50.0, 1e9], "use_bkg": True})
    assert OutlineFinder.for_session(no_background) is None, \
        "a background threshold means nothing without the background"
