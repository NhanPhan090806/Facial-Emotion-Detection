"""OpenCV's bundled Haar face detector: no external detector download needed."""

import cv2


class FaceDetector:
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
