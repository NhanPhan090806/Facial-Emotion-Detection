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
from dataclasses import replace
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
from src.data import CLASS_NAMES, build_data, make_transform, preprocessing_for
from src.model import EmotionCNN, build_resnet18, build_model
from src.predict import EmotionPredictor, load_checkpoint
from src.train import TrainConfig, fit
from src.experiments import accuracy_recipes, run_accuracy_experiments, select_candidate, verify_split


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

    def test_ddam_preprocessing_split_and_gradient(self):
        data = build_data(self.data_root, batch_size=8, val_fraction=0.25,
                          architecture="ddamfn", augmentation="fer_moderate")
        cnn = build_data(self.data_root, batch_size=8, val_fraction=0.25)
        self.assertEqual(data.val.samples, cnn.val.samples)
        self.assertEqual(data.train.samples, cnn.train.samples)
        image = Image.fromarray(np.arange(2304, dtype=np.uint8).reshape(48, 48))
        spec = preprocessing_for("ddamfn")
        mean = torch.tensor(spec["mean"])[:, None, None]
        std = torch.tensor(spec["std"])[:, None, None]
        for training in [False, True]:
            transform = make_transform(training, "fer_moderate", architecture="ddamfn")
            value = transform(image)
            self.assertEqual(tuple(value.shape), (3, 112, 112))
            restored = value * std + mean
            torch.testing.assert_close(restored[0], restored[1])
            torch.testing.assert_close(restored[1], restored[2])
        reference = make_transform(architecture="ddamfn")(image)
        torch.testing.assert_close(reference, make_transform(False, "fer_moderate", "ddamfn")(image))
        model = build_model("transfer", pretrained=False).train()
        logits, heads = model.forward_with_attention(torch.randn(2, 3, 112, 112))
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(len(heads), 2)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.features[0].conv.weight.grad).all())
        self.assertTrue(torch.isfinite(model.cat_head0.CoordAtt.conv1.weight.grad).all())

    def test_legacy_resnet_checkpoint_and_preprocessing_rejection(self):
        model = build_resnet18(pretrained=False).eval()
        path = self.work / "legacy_resnet.pt"
        checkpoint = {"format_version": 1, "model_name": "transfer", "class_names": CLASS_NAMES,
                      "preprocessing": preprocessing_for("resnet18"), "model_state": model.state_dict()}
        torch.save(checkpoint, path)
        restored, metadata = load_checkpoint(path)
        self.assertEqual(metadata["architecture"], "resnet18")
        with torch.inference_mode():
            x = torch.randn(2, 1, 48, 48)
            torch.testing.assert_close(restored(x), model(x))
        checkpoint["architecture"] = "ddamfn"
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            load_checkpoint(path)

    @unittest.skipUnless((ROOT / "tmp/ddamfn_upstream/affectnet7.pth").is_file(),
                         "Optional verified upstream checkpoint is not cached locally")
    def test_real_ddam_pretraining_mapping_and_offline_reload(self):
        from src.ddamfn import CLASS_ROW_ORDER, PRETRAINED_SHA256, download_pretrained
        path = ROOT / "tmp/ddamfn_upstream/affectnet7.pth"
        model = build_model("transfer", pretrained=True, pretrained_path=path).eval()
        official = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        torch.testing.assert_close(model.fc.weight, official["fc.weight"][CLASS_ROW_ORDER])
        torch.testing.assert_close(model.features[0].conv.weight, official["features.0.conv.weight"])
        self.assertEqual(model.pretrained_source["sha256"], PRETRAINED_SHA256)
        checkpoint = {"format_version": 2, "model_name": "transfer", "architecture": "ddamfn",
                      "class_names": CLASS_NAMES, "preprocessing": preprocessing_for("ddamfn"),
                      "model_state": model.state_dict(), "pretrained_source": model.pretrained_source}
        saved = self.work / "pretrained_ddam.pt"
        torch.save(checkpoint, saved)
        with patch("src.ddamfn.download_pretrained", side_effect=AssertionError("Inference downloaded weights")):
            restored, metadata = load_checkpoint(saved)
        x = torch.randn(2, 3, 112, 112)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), restored(x))
        # Exercise the real pretrained initialization, weighted objective and LR ramp together.
        data = build_data(self.data_root, batch_size=14, val_fraction=0.25,
                          architecture="ddamfn", augmentation="fer_moderate")
        config = TrainConfig(model_name="transfer", architecture="ddamfn", epochs=3,
                             warmup_epochs=2, pretrained=True, pretrained_path=str(path),
                             augmentation="fer_moderate", learning_rate=1e-4, backbone_lr=1e-5,
                             label_smoothing=0.02, device="cuda" if torch.cuda.is_available() else "cpu",
                             cpu_threads=2)
        with contextlib.redirect_stdout(io.StringIO()):
            run = fit(config, data, self.work / "real_pretrained_fit")
        for row in run["history"]:
            epoch = row["epoch"]
            expected_lr = 1e-5 * min(1.0, epoch / 2) * (1 + np.cos(np.pi * (epoch-1) / 3)) / 2
            self.assertAlmostEqual(row["backbone_learning_rate"], expected_lr)
            self.assertTrue(np.isfinite(row["train_objective_loss"]))
            self.assertTrue(np.isfinite(row["val_loss"]))
        self.assertGreater(sum(row["train_optimizer_steps"] for row in run["history"]), 0)
        last, _ = load_checkpoint(self.work / "real_pretrained_fit/last.pt")
        self.assertFalse(torch.equal(last.fc.weight, model.fc.weight))
        trained, metadata = load_checkpoint(run["checkpoint"])
        self.assertEqual(metadata["pretrained_source"]["sha256"], PRETRAINED_SHA256)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(trained(x)).all())
        broken = self.work / "corrupt.pth"
        broken.write_bytes(b"invalid checkpoint")
        with self.assertRaisesRegex(ValueError, "checksum"):
            download_pretrained(broken)

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
            self.assertIn("cnn_run = fit(cnn_config, data", combined)
            self.assertIn("transfer_run = fit(transfer_config, data", combined)
            self.assertNotIn("run_accuracy_experiments(", combined)
            self.assertIn("pretrained=True", combined)

    def test_accuracy_takes_priority_over_f1(self):
        rows = [{"candidate": "accurate", "val_accuracy": 0.70, "val_macro_f1": 0.60},
                {"candidate": "higher_f1", "val_accuracy": 0.69, "val_macro_f1": 0.80}]
        self.assertEqual(select_candidate(rows)["candidate"], "accurate")
        rows.append({"candidate": "tie_break", "val_accuracy": 0.70, "val_macro_f1": 0.61})
        self.assertEqual(select_candidate(rows)["candidate"], "tie_break")

    def test_augmentation_is_training_only_and_preserves_split(self):
        image = Image.fromarray(np.arange(2304, dtype=np.uint8).reshape(48, 48))
        reference = make_transform(False)(image)
        baseline = build_data(self.data_root, batch_size=8, val_fraction=0.25)
        for profile in ["lighting", "occlusion", "gentle_combined"]:
            torch.testing.assert_close(make_transform(False, profile)(image), reference)
            transformed = make_transform(True, profile)(image)
            self.assertEqual(tuple(transformed.shape), (1, 48, 48))
            self.assertTrue(torch.isfinite(transformed).all())
            self.assertGreaterEqual(transformed.min().item(), -1)
            self.assertLessEqual(transformed.max().item(), 1)
            augmented = build_data(self.data_root, batch_size=8, val_fraction=0.25, augmentation=profile)
            self.assertEqual(baseline.val.samples, augmented.val.samples)
            self.assertEqual(baseline.test.samples, augmented.test.samples)
            torch.testing.assert_close(baseline.class_weights, augmented.class_weights)
        with self.assertRaises(ValueError):
            make_transform(True, "unsupported")

    def test_reused_baseline_requires_identical_split(self):
        data = build_data(self.data_root, batch_size=8, val_fraction=0.25)
        path = self.work / "manifest_check.json"
        data.save_manifest(path)
        verify_split(path, data)
        manifest = json.loads(path.read_text())
        for split in ["train", "val"]:
            for row in manifest[split]:
                row["path"] = row["path"].replace("\\", "/")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        verify_split(path, data)
        manifest["val"][0]["path"] = "different.jpg"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "val split differs"):
            verify_split(path, data)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision_training(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for name in ["cnn", "transfer"]:
                architecture = "cnn" if name == "cnn" else "ddamfn"
                data = build_data(self.data_root, batch_size=7, val_fraction=0.25, architecture=architecture)
                config = TrainConfig(model_name=name, epochs=1, device="cuda", amp=True,
                                     pretrained=False, cpu_threads=2)
                result = fit(config, data, self.work / "cuda" / name)
                checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
                self.assertTrue(np.isfinite(checkpoint["val_metrics"]["loss"]))
                self.assertEqual(result["device"], "cuda")

    def test_optional_experiment_reuse(self):
        # The comparison helper remains supported, but is no longer in the notebooks.
        config = replace(accuracy_recipes("cnn", stage=2)["baseline"], epochs=1,
                         pretrained=False, device="cpu", cpu_threads=2)
        recipes = {"baseline": config}
        output = self.work / "optional_comparison"
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_accuracy_experiments("cnn", self.data_root, output, batch_size=8,
                                             val_fraction=0.25, recipes=recipes)
        with contextlib.redirect_stdout(io.StringIO()), patch("src.experiments.fit", side_effect=AssertionError("Unexpected retraining")):
            cached = run_accuracy_experiments("cnn", self.data_root, output, batch_size=8,
                                              val_fraction=0.25, recipes=recipes)
        self.assertEqual(cached["selected_candidate"], first["selected_candidate"])
        archive_root = self.work / "archive"
        shutil.copytree(output / "experiments/cnn/baseline", archive_root / "cnn")
        archived_config = json.loads((archive_root / "cnn/config.json").read_text())
        archived_config.pop("augmentation")
        (archive_root / "cnn/config.json").write_text(json.dumps(archived_config), encoding="utf-8")
        compact_root = self.work / "compact"
        (compact_root / "weights").mkdir(parents=True)
        (compact_root / "outputs").mkdir()
        shutil.copy2(archive_root / "cnn/best.pt", compact_root / "weights/cnn_best.pt")
        shutil.copy2(archive_root / "cnn/split_manifest.json", compact_root / "outputs/split_manifest.json")
        with contextlib.redirect_stdout(io.StringIO()), patch("src.experiments.fit", side_effect=AssertionError("Unexpected retraining")):
            imported = run_accuracy_experiments(
                "cnn", self.data_root, self.work / "imported", batch_size=8,
                val_fraction=0.25, baseline_results_root=archive_root, recipes=recipes,
                reference_results_roots={"prior": archive_root, "compact": compact_root / "outputs"},
            )
        self.assertEqual(imported["selected_candidate"], "baseline")
        self.assertEqual([row["candidate"] for row in imported["candidates"]],
                         ["baseline", "reference_prior", "reference_compact"])
        staged = self.work / "imported/experiments/cnn/reference_compact"
        self.assertFalse((staged / "history.csv").exists())
        self.assertFalse(json.loads((staged / "training_summary.json").read_text())["history_available"])

    def test_full_notebook_pipeline_and_app(self):
        """Execute local notebook code cells with tiny offline training overrides."""
        notebook = nbformat.read(ROOT / "notebooks/train.ipynb", as_version=4)
        namespace = {"__name__": "notebook_smoke_test"}
        log = io.StringIO()

        with contextlib.redirect_stdout(log), patch("IPython.display.display"):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if "DATA_ROOT = PROJECT_ROOT" in source:
                    source = (f"DATA_ROOT = Path({str(self.data_root)!r})\n"
                              f"OUTPUT_ROOT = Path({str(self.work / 'outputs')!r})\n"
                              f"WEIGHTS_DIR = Path({str(self.work / 'weights')!r})\n"
                              f"REPORT_PATH = Path({str(self.work / 'report.pdf')!r})\n"
                              "BATCH_SIZE = 8\nNUM_WORKERS = 0\nVAL_FRACTION = 0.25\nSEED = 42\n"
                              "")
                source = source.replace("epochs=80", "epochs=1").replace("epochs=40", "epochs=1")
                source = source.replace("pretrained=True", "pretrained=False").replace("warmup_epochs=2", "warmup_epochs=0")
                source = source.replace('device="auto"', 'device="cpu"')
                exec(compile(source, "notebooks/train.ipynb", "exec"), namespace)
        epoch_lines = [line for line in log.getvalue().splitlines() if line.startswith("Epoch ")]
        self.assertEqual(len(epoch_lines), 2)  # One fit per model; no candidates or per-batch output.
        for name in ["cnn", "transfer"]:
            folder = self.work / "outputs" / name
            self.assertTrue((folder / "best.pt").is_file())
            self.assertTrue((folder / "last.pt").is_file())
            self.assertTrue((folder / "training_curves.png").is_file())
            self.assertTrue((folder / "misclassified.png").is_file())
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(metrics["test_images"], 21)
            self.assertGreater(metrics["cpu_model_fps"], 0)
            self.assertTrue((folder / "validation_predictions.csv").is_file())
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
        transfer_predictor = EmotionPredictor(self.work / "weights/transfer_best.pt")
        transfer_result = transfer_predictor.predict(face)
        self.assertAlmostEqual(sum(transfer_result["probabilities"]), 1.0, places=6)
        self.assertEqual(tuple(transfer_predictor.transform(Image.fromarray(face)).shape), (3, 112, 112))
        with torch.inference_mode():
            expected = transfer_predictor.model(make_transform(architecture="ddamfn")(
                Image.fromarray(face[:, :, 0])).unsqueeze(0)).softmax(1)[0]
        torch.testing.assert_close(torch.tensor(transfer_result["probabilities"]), expected)
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
