"""CPU face detection using pretrained YuNet, with optional legacy Haar.

YuNet weights and their MIT license are bundled in assets/. No network access,
extra Python package, or emotion-model retraining is needed during inference.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

YUNET_MODEL = Path(__file__).resolve().parents[1] / "assets" / "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class YuNetFaceDetector:
    def __init__(self, model_path=None, input_size=320, score_threshold=0.75,
                 nms_threshold=0.3, min_size=40, margin=0.10):
        if input_size < 160 or input_size % 32:
            raise ValueError("YuNet input_size must be a multiple of 32 and at least 160.")
        if not 0 < score_threshold <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError("YuNet score/NMS thresholds must be in (0, 1].")
        if min_size < 1 or margin < 0:
            raise ValueError("Invalid minimum face size or crop margin.")
        path = Path(model_path) if model_path else YUNET_MODEL
        if not path.is_file():
            raise FileNotFoundError(f"YuNet model missing: {path}. Restore the bundled assets/ folder.")
        if model_path is None and hashlib.sha256(path.read_bytes()).hexdigest() != YUNET_SHA256:
            raise ValueError("Bundled YuNet model checksum mismatch; restore assets/ from the repository.")
        # Passing model bytes also supports project paths containing Unicode on Windows.
        model_bytes = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        self.network = cv2.FaceDetectorYN.create(
            "onnx", model_bytes, np.empty(0, dtype=np.uint8), (input_size, input_size),
            score_threshold, nms_threshold, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        self.input_size = input_size
        self.min_size = min_size
        self.margin = margin

    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("The face detector expects a non-empty BGR image.")
        height, width = frame.shape[:2]
        # Letterbox to a fixed square: preserve face proportions on landscape/portrait streams.
        scale = self.input_size / max(height, width)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(frame, (resized_w, resized_h))
        left = (self.input_size - resized_w) // 2
        top = (self.input_size - resized_h) // 2
        image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        image[top:top + resized_h, left:left + resized_w] = resized
        _, found = self.network.detect(image)
        if found is None:
            return []
        boxes = []
        for face in found:
            x, y, w, h = face[:4]
            if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
                continue
            # Use actual resized dimensions to undo integer rounding precisely.
            x, w = (x - left) * width / resized_w, w * width / resized_w
            y, h = (y - top) * height / resized_h, h * height / resized_h
            if min(w, h) < self.min_size:
                continue
            x1 = max(0, min(width, int(np.floor(x - w * self.margin))))
            y1 = max(0, min(height, int(np.floor(y - h * self.margin))))
            x2 = max(0, min(width, int(np.ceil(x + w * (1 + self.margin)))))
            y2 = max(0, min(height, int(np.ceil(y + h * (1 + self.margin)))))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes


class HaarFaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=40,
                 max_detection_width=640, margin=0.10):
        if scale_factor <= 1 or min_neighbors < 0 or min_size < 1 or max_detection_width < 1 or margin < 0:
            raise ValueError("Invalid face detector settings.")
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.cascade.empty():
            raise RuntimeError("Could not load OpenCV's frontal-face Haar cascade.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.max_width = max_detection_width
        self.margin = margin

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / width)
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        minimum = max(10, round(self.min_size * scale))
        found = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(minimum, minimum),
        )
        boxes = []
        for x, y, w, h in found:
            x, y, w, h = [int(round(value / scale)) for value in (x, y, w, h)]
            pad_x, pad_y = int(w * self.margin), int(h * self.margin)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes


class FaceDetector:
    """Keep the same detect(frame) -> [(x1,y1,x2,y2), ...] interface for the app."""
    def __init__(self, backend="yunet", model_path=None, input_size=320,
                 score_threshold=0.75, nms_threshold=0.3, min_size=40, margin=0.10,
                 max_detection_width=640, scale_factor=1.1, min_neighbors=5):
        self.backend = backend
        if backend == "yunet":
            self.detector = YuNetFaceDetector(model_path, input_size, score_threshold,
                                             nms_threshold, min_size, margin)
        elif backend == "haar":
            self.detector = HaarFaceDetector(scale_factor, min_neighbors, min_size,
                                             max_detection_width, margin)
        else:
            raise ValueError("Choose detector 'yunet' or 'haar'.")

    def detect(self, frame):
        return self.detector.detect(frame)
