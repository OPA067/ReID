"""
Entry point script for person Re-ID inference.

Supports processing a single image pair (target + candidate scene) and saving
the annotated result. Video support can be added by iterating over frames in
a similar fashion.
"""

import cv2
from module_reid import ReIDInfer

# -------------------------------------------------------------------------
# Initialize the ReID inference engine (loads detection + retrieval models)
# -------------------------------------------------------------------------
reid_model = ReIDInfer()

# Run inference on a target image and a candidate scene image.
# status   : bool   – True if a confident match (similarity > 0.8) is found
# output   : dict   – annotated image, similarity scores, bounding boxes, etc.
# [x1..y2]: list   – coordinates of the best-matching bounding box
# sim      : float  – similarity score of the best match
status, output, [x1, y1, x2, y2], sim = reid_model("input.jpg", "input.jpg")

# Save the annotated BGR image to disk.
cv2.imwrite("output.jpg", output["result_img"])

# (TODO) Video support: loop over frames and call reid_model(frame_target, frame_scene)
