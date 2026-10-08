"""Write a background image with the dark ring at the foot of the tank wall
painted out, for idtracker.ai's "custom" background option.

    python scripts/paint_out_ring.py <video> <output.png>

The background is the per-pixel median of 120 frames, as idtracker.ai makes
it. The ring is the dark line around the edge of the bright tank floor; it is
set to white. Reads the video only.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

video, output = Path(sys.argv[1]), Path(sys.argv[2])
cap = cv2.VideoCapture(str(video))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def gray(index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)


def ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


stack = np.stack([gray(i) for i in np.linspace(0, n - 1, 120)])
median = np.median(stack, axis=0).astype(np.uint8)
height, width = median.shape

# The floor is the bright region around the centre of the brightest-ever image,
# which has no fish in it.
bright = (cv2.GaussianBlur(stack.max(axis=0), (0, 0), 3) > 215).astype(np.uint8)
_, labels = cv2.connectedComponents(bright)
floor = cv2.morphologyEx((labels == labels[height // 2, width // 2]).astype(np.uint8),
                         cv2.MORPH_CLOSE, ellipse(41))
outline = max(cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
              key=cv2.contourArea)
floor = np.zeros_like(floor)
cv2.drawContours(floor, [outline], -1, 1, -1)

# The ring: whatever is dark within a band straddling the floor's edge.
band = cv2.dilate(floor, ellipse(61)) - cv2.erode(floor, ellipse(21))
ring = cv2.dilate(((band == 1) & (median < 185)).astype(np.uint8), ellipse(7))

painted = median.copy()
painted[ring == 1] = 255
cv2.imencode(".png", painted)[1].tofile(str(output))
print(f"{output}  {width}x{height}, ring is {ring.mean() * 100:.1f}% of the frame")
