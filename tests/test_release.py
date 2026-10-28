"""Verify the selected models, retained real results and portable project layout."""

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from src.predict import EmotionPredictor

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_selected_models_match_manifest_and_default_configuration(self):
        torch.set_num_threads(2)
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["models"]), {"cnn", "transfer"})
        for name, metadata in manifest["models"].items():
            with self.subTest(model=name):
                path = ROOT / config["models"][name]
                self.assertEqual(path.parent, ROOT / "models")
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
                predictor = EmotionPredictor(path, device="cpu")
                self.assertEqual(predictor.checkpoint["architecture"], metadata["architecture"])
                self.assertEqual(predictor.checkpoint["epoch"], metadata["best_epoch"])
                result = predictor.predict(np.full((80, 80, 3), 128, dtype=np.uint8))
                self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=6)

    def test_retained_test_predictions_match_measured_scores(self):
        self.assertTrue((ROOT / "reports/report.pdf").is_file())
        self.assertTrue((ROOT / "docs/assignment.pdf").is_file())
        self.assertTrue((ROOT / "reports/training.log").is_file())
        root = ROOT / "reports/results"
        self.assertFalse(list(root.rglob("*.pt")))
        for name in ["cnn", "transfer"]:
            folder = root / name
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            with (folder / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), metrics["test_images"])
            correct = sum(row["correct"].lower() == "true" for row in rows)
            self.assertAlmostEqual(correct / len(rows), metrics["test_accuracy"])
            for filename in ["training_curves.png", "confusion_matrix.png", "misclassified.png"]:
                self.assertTrue((folder / filename).is_file())

    def test_markdown_is_portable_utf8(self):
        for path in ROOT.rglob("*.md"):
            if any(part in {".git", "tmp"} for part in path.relative_to(ROOT).parts):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_venv", text, str(path))
            self.assertNotIn("C:" + chr(92) + "Users", text, str(path))
            self.assertNotIn(chr(0xfffd), text, str(path))
        # The root README uses an ASCII tree, avoiding the previous corrupted box drawing.
        self.assertTrue((ROOT / "README.md").read_text(encoding="utf-8").isascii())


if __name__ == "__main__":
    unittest.main()
