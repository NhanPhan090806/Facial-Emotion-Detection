"""Face-detector geometry and camera-label regression tests (no network needed)."""

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.app import draw_emotion_label, draw_fps
from src.detector import FaceDetector, YuNetFaceDetector, YUNET_MODEL

ROOT = Path(__file__).resolve().parents[1]


class DetectorTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)

    def test_bundled_yunet_loads_and_blank_has_no_faces(self):
        self.assertTrue(YUNET_MODEL.is_file())
        detector = FaceDetector()
        self.assertEqual(detector.backend, "yunet")
        self.assertEqual(detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)), [])
        self.assertEqual(FaceDetector(backend="haar").backend, "haar")

    def test_letterbox_landscape_and_portrait_coordinates(self):
        for shape, detection, expected in [
            ((360, 640, 3), [100, 115, 60, 60], (200, 90, 320, 210)),
            ((640, 360, 3), [115, 100, 60, 60], (90, 200, 210, 320)),
        ]:
            with self.subTest(shape=shape), patch("src.detector.cv2.FaceDetectorYN.create") as create:
                create.return_value.detect.return_value = (1, np.array([detection + [0]*10 + [.95]], dtype=np.float32))
                detector = YuNetFaceDetector(margin=0)
                boxes = detector.detect(np.zeros(shape, dtype=np.uint8))
                self.assertEqual(boxes, [expected])
                supplied = create.return_value.detect.call_args.args[0]
                self.assertEqual(supplied.shape, (320, 320, 3))

    def test_boxes_are_clipped_and_small_or_invalid_detections_skipped(self):
        detections = [
            [-10, 55, 80, 80], [300, 220, 80, 80],
            [100, 100, 5, 5], [0, 0, -5, 50], [float("nan"), 10, 80, 80],
        ]
        found = np.array([row + [0]*10 + [.9] for row in detections], dtype=np.float32)
        with patch("src.detector.cv2.FaceDetectorYN.create") as create:
            create.return_value.detect.return_value = (1, found)
            detector = YuNetFaceDetector()
            boxes = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
            self.assertEqual(len(boxes), 2)
            for x1, y1, x2, y2 in boxes:
                self.assertTrue(0 <= x1 < x2 <= 640)
                self.assertTrue(0 <= y1 < y2 <= 360)
            self.assertEqual(boxes[0][:2], (0, 0))
            self.assertEqual(boxes[1][2:], (640, 360))
            create.return_value.detect.return_value = (1, None)
            self.assertEqual(detector.detect(np.zeros((360, 640, 3), dtype=np.uint8)), [])
            with self.assertRaises(ValueError):
                detector.detect(np.zeros((10, 10), dtype=np.uint8))
        for kwargs in [{"input_size": 300}, {"score_threshold": 0}, {"margin": -1}]:
            with self.assertRaises(ValueError):
                YuNetFaceDetector(**kwargs)

    @unittest.skipUnless((ROOT / "tmp/yunet_qa/lena.jpg").is_file(), "Optional public OpenCV sample is not cached")
    def test_real_sample_face_detected(self):
        image = cv2.imdecode(np.fromfile(ROOT / "tmp/yunet_qa/lena.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
        boxes = FaceDetector().detect(image)
        self.assertGreaterEqual(len(boxes), 1)
        self.assertTrue(any(x1 < 280 < x2 and y1 < 280 < y2 for x1,y1,x2,y2 in boxes))

    def test_labels_are_large_readable_and_stay_inside_frame(self):
        confident = {"label": "surprise", "confidence": .94}
        for shape in [(480, 640, 3), (720, 1280, 3), (96, 96, 3)]:
            height, width = shape[:2]
            for box in [(0, 0, 40, 50), (width-50, height-60, width-1, height-1)]:
                frame = np.full(shape, 180, dtype=np.uint8)
                with patch("src.app.cv2.putText", wraps=cv2.putText) as draw:
                    bounds = draw_emotion_label(frame, box, confident, .4)
                x1,y1,x2,y2 = bounds
                self.assertTrue(0 <= x1 < x2 < width and 0 <= y1 < y2 < height)
                self.assertEqual(draw.call_args_list[0].args[1], "SURPRISE")
                self.assertEqual(draw.call_args_list[1].args[1], "94.0% confidence")
                if height >= 480:
                    self.assertGreater(draw.call_args_list[0].args[4], .55)
                # A solid panel protects text from bright or detailed backgrounds.
                self.assertTrue(np.all(frame[y1+2, x1+2] == 18))
        frame = np.zeros((480,640,3), dtype=np.uint8)
        with patch("src.app.cv2.putText", wraps=cv2.putText) as draw:
            draw_emotion_label(frame, (20, 100, 200, 300), {"label":"fear","confidence":.31}, .4)
        self.assertEqual(draw.call_args_list[0].args[1], "UNCERTAIN")
        self.assertIn("fear", draw.call_args_list[1].args[1])
        draw_fps(frame, 12.3, 2)
        self.assertTrue(np.any(frame[-40:]))


if __name__ == "__main__":
    unittest.main()
