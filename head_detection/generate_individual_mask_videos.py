import cv2
import numpy as np
from idtrackerai import ListOfBlobs

SESSION_DIR = "session_batch1control_freeswim_2026-03-27-141723-good"
SOURCE_VIDEO = (
    r"C:\Users\grich\Macquarie University\Morsch Group - Documents\Grant R"
    r"\Fish videos\Pradeep Fish Free Swim Videos"
    r"\27032026 Pradeep's Fish Free Swim\batch1control_freeswim_2026-03-27-141723-good.avi"
)
OUTPUT_DIR = "individual_mask_videos"
N_FRAMES = 500
CROP_SIZE = 158  # match idtrackerai individual video size

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load blobs
print("Loading blobs...")
list_of_blobs = ListOfBlobs.load(f"{SESSION_DIR}/preprocessing/list_of_blobs.pickle")
n_frames = min(N_FRAMES, len(list_of_blobs.blobs_in_video))
print(f"Loaded blobs for {len(list_of_blobs.blobs_in_video)} frames, processing {n_frames}")

# Get frame size and FPS from source video
cap = cv2.VideoCapture(SOURCE_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Frame size: {frame_width}x{frame_height}, FPS: {fps:.2f}")

# Discover all identities present in the first N frames
identities = set()
for fn in range(n_frames):
    for blob in list_of_blobs.blobs_in_video[fn]:
        if not blob.is_a_crossing:
            identities.add(blob.identity)
identities = sorted(i for i in identities if i is not None)
print(f"Identities found: {identities}")

half = CROP_SIZE // 2
fourcc = cv2.VideoWriter_fourcc(*"XVID")

writers = {
    identity: cv2.VideoWriter(
        f"{OUTPUT_DIR}/fish_{identity}_mask.avi",
        fourcc, fps, (CROP_SIZE, CROP_SIZE), isColor=False
    )
    for identity in identities
}

# Build full-frame mask once per frame, then crop per identity
for frame_number in range(n_frames):
    if frame_number % 50 == 0:
        print(f"  Frame {frame_number}/{n_frames}...")

    blobs_in_frame = list_of_blobs.blobs_in_video[frame_number]

    # Index blobs by identity for this frame
    blob_by_identity = {}
    for blob in blobs_in_frame:
        if not blob.is_a_crossing and blob.identity is not None:
            blob_by_identity[blob.identity] = blob

    for identity in identities:
        if identity not in blob_by_identity:
            writers[identity].write(np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8))
            continue

        blob = blob_by_identity[identity]
        cx, cy = blob.centroid  # (x, y) = (col, row)
        cx, cy = int(round(cx)), int(round(cy))

        bc = blob.bbox_corners
        x_min, y_min = bc.bottom, bc.left
        blob_mask = blob.get_bbox_mask()  # shape (h, w), values 0 or 1
        bh, bw = blob_mask.shape

        # Paint only this blob's mask onto a blank CROP_SIZE canvas.
        # The canvas is centered on (cx, cy) in frame coords.
        canvas = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

        # Where does the blob bbox start in canvas coords?
        canvas_col = x_min - (cx - half)
        canvas_row = y_min - (cy - half)

        # Compute the overlapping region between blob bbox and canvas
        src_r0 = max(0, -canvas_row)
        src_c0 = max(0, -canvas_col)
        dst_r0 = max(0, canvas_row)
        dst_c0 = max(0, canvas_col)
        copy_h = min(bh - src_r0, CROP_SIZE - dst_r0)
        copy_w = min(bw - src_c0, CROP_SIZE - dst_c0)

        if copy_h > 0 and copy_w > 0:
            region = blob_mask[src_r0:src_r0 + copy_h, src_c0:src_c0 + copy_w]
            canvas[dst_r0:dst_r0 + copy_h, dst_c0:dst_c0 + copy_w] = np.where(
                region > 0, 255, 0
            )

        writers[identity].write(canvas)

for w in writers.values():
    w.release()

print(f"Done! Videos saved to {OUTPUT_DIR}/")
