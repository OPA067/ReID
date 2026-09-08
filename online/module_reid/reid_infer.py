"""
Person Re-Identification (ReID) inference module.

Provides a high-level interface that combines YOLO-based person detection
with a CLIP-based image encoder to perform target-person retrieval in a
candidate scene image.

Usage:
    from reid_module import ReIDInfer

    infer = ReIDInfer()
    output = infer("target.jpg", "scene.jpg")   # output["result_img"] is annotated BGR
    print("Best similarity:", output["best_sim"])
"""

import cv2
import os
import numpy as np
import torch

from PIL import Image
import torchvision.transforms as T
from .utils.checkpoint import Checkpointer
from .utils.iotools import load_train_configs
from .model import build_model
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

# Absolute paths to default model artifacts shipped inside the package.
_PACKAGE_DIR    = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_PACKAGE_DIR, "experiments/20260601_141555_baseline_reid/configs.yaml")
_DEFAULT_WEIGHT = os.path.join(_PACKAGE_DIR, "experiments/20260601_141555_baseline_reid/best.pth")
_DEFAULT_YOLO   = os.path.join(_PACKAGE_DIR, "experiments/20260601_141555_baseline_reid/yolov8l.pt")

# ImageNet-normalized CLIP mean / std (used for input preprocessing).
_mean = [0.48145466, 0.4578275, 0.40821073]
_std = [0.26862954, 0.26130258, 0.27577711]
_transform = T.Compose([
    T.Resize((384, 128)),      # (H, W) – ReID-standard aspect ratio
    T.ToTensor(),
    T.Normalize(mean=_mean, std=_std),
])

def _resolve_data_path(path):
    """Resolve a potentially relative data path against the package directory."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    pkg_path = os.path.join(_PACKAGE_DIR, path)
    if os.path.exists(pkg_path):
        return pkg_path
    return path

class ReIDInfer:
    """Person ReID client. Loads detection & retrieval models once, then call on image pairs."""

    def __init__(self, ):

        self.config_file = _DEFAULT_CONFIG
        self.weight_file = _DEFAULT_WEIGHT
        self.yolo_model  = _DEFAULT_YOLO

        # Select compute device (prefer GPU when available).
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build model configuration from YAML.
        args = load_train_configs(self.config_file)
        args.training = False

        # Load the image-encoder (retrieval) model and restore trained weights.
        self.retrieval_model = build_model(args)
        checkpointer = Checkpointer(self.retrieval_model)
        checkpointer.load(self.weight_file)
        self.retrieval_model = self.retrieval_model.to(self.device)
        self.retrieval_model.eval()

        # Load the YOLO detection model.
        self.detect_model = YOLO(self.yolo_model)

        # Simple cache for target-image feature vectors
        # (cache key = resolved path; skipped for PIL.Image inputs).
        self._target_cache_key  = None
        self._target_cache_feat = None

    @staticmethod
    def _compute_feat(model, image):
        """Extract L2-normalized feature embedding from the retrieval model."""
        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def _load_target(self, target_image):
        """
        Load and encode the target person image.

        Args:
            target_image: file path (str/bytes) or PIL.Image.

        Returns:
            L2-normalized feature tensor of shape (1, embed_dim).
        """
        # Determine cache key (None for PIL.Image -> skip caching).
        if isinstance(target_image, (str, bytes)):
            cache_key = _resolve_data_path(target_image)
        else:
            cache_key = None

        # Return cached feature if the same target is requested again.
        if cache_key is not None and cache_key == self._target_cache_key:
            return self._target_cache_feat

        # Open image, apply transforms, and move to compute device.
        if isinstance(target_image, (str, bytes)):
            target_image = Image.open(_resolve_data_path(target_image)).convert("RGB")
        t = _transform(target_image).unsqueeze(0).to(self.device)
        feat = self._compute_feat(self.retrieval_model, t)

        # Update cache.
        if cache_key is not None:
            self._target_cache_key  = cache_key
            self._target_cache_feat = feat

        return feat

    def __call__(self, target_image, candidate_image, conf_thresh=0.5):
        """
        Run full person-search pipeline: detect persons -> encode -> compare against target.

        Args:
            target_image:   path (str) or PIL.Image of the target person.
            candidate_image: path (str) or BGR numpy array of the scene image.
            conf_thresh:    minimum detection confidence for a box to be considered.

        Returns:
            Tuple (status, out_dict, best_box, best_sim):
                status   : bool  – True if the best match exceeds similarity threshold (0.8)
                out_dict : dict  – contains:
                    - result_img: annotated BGR image (numpy array)
                    - best_sim  : similarity score of the best match
                    - best_box  : bounding box of the best match (x1, y1, x2, y2)
                    - boxes     : list of all detected boxes above conf_thresh
                    - scores    : list of similarity scores for each detected box
                best_box : list  – [x1, y1, x2, y2] or [None, None, None, None]
                best_sim : float – best similarity score or 0
        """
        # ---- 1. Encode target person ----
        tar_feat = self._load_target(target_image)

        # ---- 2. Load candidate (scene) image ----
        if isinstance(candidate_image, str):
            img_bgr = cv2.imread(_resolve_data_path(candidate_image))
        else:
            img_bgr = candidate_image.copy()
        if img_bgr is None:
            raise ValueError(f"Cannot read candidate image: {candidate_image}")

        # ---- 3. Detect persons with YOLO ----
        results = self.detect_model.predict(img_bgr, device=self.device, classes=[0], verbose=False)
        detections = results[0].boxes.data

        # Crop each detected person and prepare for the retrieval model.
        crops, boxes = [], []
        for det in detections:
            x1, y1, x2, y2, conf, _ = det.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if conf > conf_thresh:
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_t = _transform(crop_pil)
                crops.append(crop_t)
                boxes.append((x1, y1, x2, y2))

        out = {
            "result_img": img_bgr,
            "best_sim": None,
            "best_box": None,
            "boxes": boxes,
            "scores": [],
        }

        # Early exit if no persons detected.
        if not crops:
            return False, out, [None, None, None, None], 0

        # ---- 4. Compute similarity between target and all detected crops ----
        crops_t = torch.stack(crops).to(self.device)
        can_feat = self._compute_feat(self.retrieval_model, crops_t)
        sims = (tar_feat @ can_feat.T).squeeze(0).cpu().numpy()
        out["scores"] = sims.tolist()

        # ---- 5. Identify best match and annotate image ----
        best_idx = int(np.argmax(sims))
        x1, y1, x2, y2 = boxes[best_idx]
        best_sim = float(sims[best_idx])
        out["best_box"] = (x1, y1, x2, y2)
        out["best_sim"] = best_sim
        if best_sim > 0.8:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"sims_{best_sim:.4f}"
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Confident match found: return box coordinates for downstream control.
            return True, out, [x1, y1, x2, y2], best_sim
        else:
            return False, out, [None, None, None, None], 0