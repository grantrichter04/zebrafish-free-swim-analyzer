import cv2
import numpy as np
from idtrackerai import ListOfBlobs

SESSION_DIR = "session_batch1control_freeswim_2026-03-27-141723-good"
SOURCE_VIDEO = (
    r"C:\Users\grich\Macquarie University\Morsch Group - Documents\Grant R"
    r"\Fish videos\Pradeep Fish Free Swim Videos"
    r"\27032026 Pradeep's Fish Free Swim\batch1control_freeswim_2026-03-27-141723-good.avi"
)
OUTPUT_VIDEO = "mask_video_500frames.avi"
N_FRAMES = 500

# Load blobs
print("Loading blobs...")
list_of_blobs = ListOfBlobs.load(f"{SESSION_DIR}/preprocessing/list_of_blobs.pickle")
print(f"Loaded blobs for {len(list_of_blobs.blobs_in_video)} frames")

# Get frame size from source video
cap = cv2.VideoCapture(SOURCE_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Frame size: {frame_width}x{frame_height}, FPS: {fps}")

# Set up video writer (grayscale: isColor=False)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height), isColor=False)

n_frames = min(N_FRAMES, len(list_of_blobs.blobs_in_video))

for frame_number in range(n_frames):
    if frame_number % 50 == 0:
        print(f"  Processing frame {frame_number}/{n_frames}...")

    frame_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    blobs_in_frame = list_of_blobs.blobs_in_video[frame_number]
    for blob in blobs_in_frame:
        # bbox_corners: BoundingBoxCoordinates(bottom=x_min, left=y_min, top=x_max, right=y_max)
        bc = blob.bbox_corners
        x_min, y_min = bc.bottom, bc.left
        blob_mask = blob.get_bbox_mask()  # binary, shape (h, w), values 0 or 1

        h, w = blob_mask.shape
        # Clip to frame bounds
        r0 = max(y_min, 0)
        r1 = min(y_min + h, frame_height)
        c0 = max(x_min, 0)
        c1 = min(x_min + w, frame_width)

        mr0 = r0 - y_min
        mr1 = mr0 + (r1 - r0)
        mc0 = c0 - x_min
        mc1 = mc0 + (c1 - c0)

        frame_mask[r0:r1, c0:c1] = np.where(
            blob_mask[mr0:mr1, mc0:mc1] > 0, 255, frame_mask[r0:r1, c0:c1]
        )

    out.write(frame_mask)

out.release()
print(f"Done! Saved to {OUTPUT_VIDEO}")
