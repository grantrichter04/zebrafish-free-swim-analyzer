"""Score ways of turning a frame into a mask of fish, on real videos.

    python scripts/score_segmentation.py <folder of videos> <number of fish>

For each .avi in the folder and each method, it reports the share of 150
sampled frames in which the mask has exactly that many separate blobs, at the
method's best threshold. idtracker.ai learns identities from frames where
every animal is its own blob, so that share is the honest score for a
segmentation - and for a lighting setup.

The first method is what idtracker.ai itself does. The rest are candidates for
an opt-in replacement; see docs/handover-2026-10-08.md. Reads the videos only.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

folder = Path(sys.argv[1])
N_FISH = int(sys.argv[2])
MIN_AREA = 150
ELLIPSE5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ELLIPSE31 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def count(mask):
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    return sum(1 for c in contours if cv2.contourArea(c) >= MIN_AREA)


for video in sorted(folder.glob("*.avi")):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def gray(index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

    stack = np.stack([gray(i) for i in np.linspace(0, n - 1, 50)])
    median = np.median(stack, axis=0).astype(np.uint8)
    median_f = np.maximum(median, 1).astype(np.float32)
    frames = [gray(i) for i in np.linspace(5, n - 6, 150)]

    def relative(f):
        return cv2.GaussianBlur(
            np.clip((median_f - f) / median_f * 255, 0, 255).astype(np.uint8), (0, 0), 1.5)

    # image-producing step, then an optional clean-up of the thresholded mask
    methods = {
        "idtracker.ai: |frame - bkg|": (lambda f: cv2.absdiff(median, f), None),
        "relative to bkg, blurred": (relative, None),
        "  + open (cut thin shadow bridges)": (
            relative, lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, ELLIPSE5)),
        "  + close (join a fish in pieces)": (
            relative, lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, ELLIPSE5)),
        "black-hat, no background at all": (
            lambda f: cv2.morphologyEx(cv2.GaussianBlur(f, (0, 0), 1.5),
                                       cv2.MORPH_BLACKHAT, ELLIPSE31), None),
        "relative x black-hat (both agree)": (
            lambda f: np.sqrt(relative(f).astype(np.float32) * cv2.morphologyEx(
                cv2.GaussianBlur(f, (0, 0), 1.5), cv2.MORPH_BLACKHAT, ELLIPSE31)
            ).astype(np.uint8), None),
    }
    print(f"\n{video.name[:44]}")
    for name, (make, clean) in methods.items():
        images = [make(f) for f in frames]
        rows = []
        for threshold in range(8, 101, 4):
            counts = []
            for image in images:
                mask = (image > threshold).astype(np.uint8)
                counts.append(count(clean(mask) if clean else mask))
            rows.append((100 * np.mean(np.array(counts) == N_FISH), threshold))
        best = max(rows)
        usable = [t for share, t in rows if share >= best[0] - 5]
        print(f"  {name:36s} {best[0]:5.1f}% at threshold {best[1]:3d} "
              f"(about as good over {min(usable)}-{max(usable)})")
