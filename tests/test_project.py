"""Offline tests using generated images, never presented as FER accuracy results.

Run: python -m unittest discover -s tests -v
"""

import ast
import csv
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import nbformat
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.models import resnet18

from src.app import annotate_frame, main
from src.data import CLASS_NAMES, build_data, make_transform
from src.model import EmotionCNN, build_resnet18
from src.predict import EmotionPredictor, load_checkpoint
from src.train import TrainConfig, fit


ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (ROOT / "tmp").mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="fer_test_", dir=ROOT / "tmp")
        cls.work = Path(cls.temp.name)
        cls.data_root = cls.work / "data"
        rng = np.random.default_rng(42)
        for split, count in [("train", 8), ("test", 3)]:
            for label, name in enumerate(CLASS_NAMES):
                folder = cls.data_root / split / name
                folder.mkdir(parents=True)
                for index in range(count):
                    pixels = rng.integers(0, 255, size=(48, 48), dtype=np.uint8)
                    Image.fromarray(pixels).save(folder / f"{label}_{index}.png")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_supplied_cnn_architecture(self):
        model = EmotionCNN().eval()
        value = torch.randn(2, 1, 48, 48)
        for block, channels, size, probability in zip(model.features, [32, 64, 128, 256],
                                                       [24, 12, 6, 3], [0.1, 0.15, 0.2, 0.25]):
            value = block(value)
            self.assertEqual(tuple(value.shape), (2, channels, size, size))
            self.assertEqual(block[-1].p, probability)
        self.assertEqual(tuple(model.classifier(value).shape), (2, 7))
        self.assertEqual(model.classifier[1].in_features, 2304)
        self.assertEqual(model.classifier[1].out_features, 256)
        self.assertEqual(model.classifier[4].p, 0.5)

    def test_pretrained_grayscale_adaptation_without_download(self):
        original = resnet18(weights=None)
        expected = original.conv1.weight.detach().sum(dim=1, keepdim=True)
        with patch("src.model.resnet18", return_value=original) as builder:
            adapted = build_resnet18(pretrained=True)
        self.assertIsNotNone(builder.call_args.kwargs["weights"])
        torch.testing.assert_close(adapted.conv1.weight, expected)
        self.assertEqual(adapted.conv1.stride, (1, 1))
        self.assertIsInstance(adapted.maxpool, torch.nn.Identity)
        self.assertEqual(adapted.fc[-1].out_features, 7)
        adapted.eval()
        with torch.inference_mode():
            self.assertEqual(tuple(adapted(torch.zeros(2, 1, 48, 48)).shape), (2, 7))

    def test_split_and_weights_do_not_use_test(self):
        data = build_data(self.work, batch_size=8, val_fraction=0.25)
        again = build_data(self.data_root, batch_size=8, val_fraction=0.25)
        self.assertEqual(data.train.samples, again.train.samples)
        self.assertEqual(data.val.samples, again.val.samples)
        train_paths = {p for p, _ in data.train.samples}
        val_paths = {p for p, _ in data.val.samples}
        test_paths = {p for p, _ in data.test.samples}
        self.assertFalse(train_paths & val_paths)
        self.assertFalse((train_paths | val_paths) & test_paths)
        self.assertEqual(len(train_paths | val_paths), 56)
        counts = np.bincount(data.train.targets, minlength=7)
        expected = np.sqrt(counts.sum() / (7 * counts)); expected /= expected.mean()
        np.testing.assert_allclose(data.class_weights.numpy(), expected, rtol=1e-6)
        image, _ = data.val[0]
        self.assertEqual(tuple(image.shape), (1, 48, 48))
        self.assertGreaterEqual(image.min().item(), -1)
        self.assertLessEqual(image.max().item(), 1)

    def test_notebook_syntax_and_shared_training(self):
        for path in [ROOT / "notebooks/train.ipynb", ROOT / "train_kaggle.ipynb"]:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
            combined = "\n".join(cell.source for cell in notebook.cells)
            self.assertIn("from src.train import", combined)
            self.assertIn("fit(cnn_config, data", combined)
            self.assertIn("fit(transfer_config, data", combined)
            self.assertIn("pretrained=True", combined)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision_training(self):
        data = build_data(self.data_root, batch_size=14, val_fraction=0.25)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("src.train.tqdm", side_effect=lambda items, **kwargs: items):
            for name in ["cnn", "transfer"]:
                config = TrainConfig(model_name=name, epochs=1, device="cuda", amp=True,
                                     pretrained=False, cpu_threads=2)
                result = fit(config, data, self.work / "cuda" / name)
                checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
                self.assertTrue(np.isfinite(checkpoint["val_metrics"]["loss"]))
                self.assertEqual(result["device"], "cuda")

    def test_full_notebook_pipeline_and_app(self):
        """Execute local notebook code cells with tiny offline training overrides."""
        notebook = nbformat.read(ROOT / "notebooks/train.ipynb", as_version=4)
        namespace = {"__name__": "notebook_smoke_test"}
        log = io.StringIO()
        with contextlib.redirect_stdout(log), patch("IPython.display.display"), \
                patch("src.train.tqdm", side_effect=lambda items, **kwargs: items):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if "DATA_ROOT = PROJECT_ROOT" in source:
                    source = (f"DATA_ROOT = Path({str(self.data_root)!r})\n"
                              f"OUTPUT_ROOT = Path({str(self.work / 'outputs')!r})\n"
                              f"WEIGHTS_DIR = Path({str(self.work / 'weights')!r})\n"
                              f"REPORT_PATH = Path({str(self.work / 'report.pdf')!r})\n"
                              "BATCH_SIZE = 8\nNUM_WORKERS = 0\nVAL_FRACTION = 0.25\nSEED = 42")
                source = source.replace("epochs=80", "epochs=1").replace("epochs=40", "epochs=1")
                source = source.replace("pretrained=True", "pretrained=False")
                source = source.replace('device="auto"', 'device="cpu"')
                exec(compile(source, "notebooks/train.ipynb", "exec"), namespace)
        for name in ["cnn", "transfer"]:
            folder = self.work / "outputs" / name
            self.assertTrue((folder / "best.pt").is_file())
            self.assertTrue((folder / "last.pt").is_file())
            self.assertTrue((folder / "training_curves.png").is_file())
            self.assertTrue((folder / "misclassified.png").is_file())
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(metrics["test_images"], 21)
            self.assertGreater(metrics["cpu_model_fps"], 0)
            with (folder / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                examples = [row for row in csv.DictReader(stream) if row["example_number"]]
            self.assertEqual(sorted(int(row["example_number"]) for row in examples), list(range(1, 11)))
            self.assertEqual(len((folder / "split_manifest.json").read_text()),
                             len((self.work / "outputs" / "split_manifest.json").read_text()))
        self.assertTrue((self.work / "report.pdf").is_file())
        if os.environ.get("FER_QA_DIR"):
            qa_dir = Path(os.environ["FER_QA_DIR"])
            qa_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.work / "report.pdf", qa_dir / "synthetic_report.pdf")
            shutil.copytree(self.work / "outputs", qa_dir / "synthetic_outputs", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pt"))
        predictor = EmotionPredictor(self.work / "weights/cnn_best.pt")
        face = np.full((60, 60, 3), 128, dtype=np.uint8)
        self.assertEqual(len(predictor.predict_faces([face, face])), 2)
        self.assertEqual(predictor.predict_faces([]), [])
        result = predictor.predict(face)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=6)
        with torch.inference_mode():
            expected = predictor.model(make_transform()(Image.fromarray(face[:, :, 0])).unsqueeze(0)).softmax(1)[0]
        torch.testing.assert_close(torch.tensor(result["probabilities"]), expected)
        # Multiple detector boxes produce multiple face classifications.
        detector = type("Detector", (), {"detect": lambda _, frame: [(0, 0, 40, 40), (20, 20, 60, 60)]})()
        _, results = annotate_frame(face.copy(), detector, predictor, 0.4)
        self.assertEqual(len(results), 2)
        config_path = self.work / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "weights": {"cnn": str(self.work / "weights/cnn_best.pt"),
                        "transfer": str(self.work / "weights/transfer_best.pt")},
            "camera": {"reconnect_attempts": 2, "reconnect_delay": 0},
            "cpu_threads": 2,
        }), encoding="utf-8")
        image_path = self.work / "blank.png"
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(image_path)
        output_path = self.work / "annotated.png"
        base = ["--config", str(config_path), "--headless"]
        self.assertEqual(main(base + ["--source", str(image_path), "--output", str(output_path)]), 0)
        self.assertTrue(output_path.is_file())
        self.assertEqual(main(base + ["--source", str(image_path), "--model", "transfer",
                                      "--output", str(self.work / "transfer.png")]), 0)
        video_path = self.work / "input.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()
        self.assertEqual(main(base + ["--source", str(video_path), "--output", str(self.work / "output.avi")]), 0)
        with patch("src.app.open_capture") as capture:
            capture.return_value.isOpened.return_value = False
            self.assertEqual(main(base + ["--source", "http://127.0.0.1:1/video"]), 2)
            self.assertEqual(capture.call_count, 3)
        # Metadata mismatches fail before inference rather than silently swapping labels.
        bad_path = self.work / "bad.pt"
        checkpoint = predictor.checkpoint.copy(); checkpoint["class_names"] = list(reversed(CLASS_NAMES))
        torch.save(checkpoint, bad_path)
        with self.assertRaisesRegex(ValueError, "class order"):
            load_checkpoint(bad_path)


if __name__ == "__main__":
    unittest.main()
