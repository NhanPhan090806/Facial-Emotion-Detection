"""Small, accuracy-first validation search. Test images never choose the winner."""

import csv
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import CLASS_NAMES, build_data
from src.predict import load_checkpoint
from src.train import TrainConfig, fit, resolve_device


def accuracy_recipes(model_name, seed=42, device="auto"):
    """Include the original recipe so new augmentation is never assumed better.

    This is a practical candidate search, not a one-factor causal ablation.
    M2's regularized candidate deliberately combines a few modest changes.
    """
    if model_name not in {"cnn", "transfer"}:
        raise ValueError("Choose cnn or transfer.")
    baseline = TrainConfig(model_name=model_name, epochs=80 if model_name == "cnn" else 20,
                           pretrained=model_name == "transfer", seed=seed, device=device,
                           patience=18 if model_name == "cnn" else 12)
    recipes = {
        "baseline": baseline,
        "lighting": replace(baseline, augmentation="lighting"),
    }
    if model_name == "transfer":
        recipes["regularized"] = replace(
            baseline, augmentation="occlusion", backbone_lr=5e-5, weight_decay=5e-4,
        )
    return recipes


def _write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _normalized_split(rows):
    # Compare Kaggle/Linux and Windows manifests using the same path separators.
    return sorted((row["path"].replace("\\", "/"), int(row["label"])) for row in rows)


def verify_split(manifest_path, data):
    """A reused model must not have trained on this run's validation images."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for split, dataset in [("train", data.train), ("val", data.val)]:
        current = [{"path": str(Path(path).relative_to(data.root)), "label": label}
                   for path, label in dataset.samples]
        if _normalized_split(manifest.get(split, [])) != _normalized_split(current):
            raise ValueError(f"{split} split differs from {manifest_path}; cannot reuse this run.")


@torch.inference_mode()
def validate_checkpoint(checkpoint_path, data, output_dir, device="auto"):
    """Full-precision validation predictions, including per-class diagnostics."""
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    truth, predictions, confidences = [], [], []
    loss_sum = 0.0
    for images, labels in data.val_loader:
        logits = model(images.to(device))
        loss_sum += torch.nn.functional.cross_entropy(
            logits, labels.to(device), reduction="sum",
        ).item()
        probabilities = logits.softmax(dim=1).cpu()
        truth.extend(labels.tolist())
        predictions.extend(probabilities.argmax(1).tolist())
        confidences.extend(probabilities.max(1).values.tolist())
    metrics = {"accuracy": float(accuracy_score(truth, predictions)),
               "macro_f1": float(f1_score(truth, predictions, labels=list(range(7)),
                                          average="macro", zero_division=0)),
               "loss": loss_sum / len(truth), "images": len(truth),
               "epoch": checkpoint["epoch"]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metrics, output_dir / "validation_metrics.json")
    _write_json(classification_report(truth, predictions, labels=list(range(7)),
                                     target_names=CLASS_NAMES, zero_division=0, output_dict=True),
                output_dir / "validation_classification_report.json")
    np.savetxt(output_dir / "validation_confusion_matrix.csv",
               confusion_matrix(truth, predictions, labels=list(range(7))), fmt="%d", delimiter=",")
    with (output_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true", "predicted", "confidence"])
        for (path, label), predicted, confidence in zip(data.val.samples, predictions, confidences):
            writer.writerow([str(Path(path).relative_to(data.root)), CLASS_NAMES[label],
                             CLASS_NAMES[predicted], confidence])
    return metrics


def select_candidate(rows):
    """Accuracy always wins; use macro F1 only when accuracy is exactly tied."""
    if not rows:
        raise ValueError("No candidates to select.")
    return max(rows, key=lambda row: (row["val_accuracy"], row["val_macro_f1"]))


def run_accuracy_experiments(model_name, data_root, output_root, batch_size=128,
                             num_workers=0, val_fraction=0.1, split_seed=42,
                             seed=42, device="auto", baseline_results_root=None,
                             recipes=None):
    """Train/cache candidates, rank ONLY on validation, export one selected model.

    Optional baseline_results_root points to an old outputs/ with model/best.pt
    and model/split_manifest.json. Missing or mismatched baselines fail explicitly.
    Completed matching experiments are reused; interrupted fits cannot resume.
    """
    recipes = accuracy_recipes(model_name, seed, device) if recipes is None else recipes
    if "baseline" not in recipes:
        raise ValueError("Include baseline in the candidate recipes.")
    output_root = Path(output_root)
    experiment_root = output_root / "experiments" / model_name
    rows, sources = [], {}
    for name, config in recipes.items():
        if config.model_name != model_name or name in {"", ".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Invalid candidate name or model configuration.")
        data = build_data(data_root, batch_size=batch_size, num_workers=num_workers,
                          val_fraction=val_fraction, seed=split_seed, augmentation=config.augmentation)
        candidate_dir = experiment_root / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        reused_baseline = name == "baseline" and baseline_results_root is not None
        if reused_baseline:
            source = Path(baseline_results_root) / model_name
            verify_split(source / "split_manifest.json", data)
            old_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            # The first run predates the augmentation key; that means baseline.
            expected = asdict(config)
            ignored = {"device", "cpu_threads", "amp"}
            if any(old_config.get(key, "baseline" if key == "augmentation" else None) != value
                   for key, value in expected.items() if key not in ignored):
                raise ValueError("The supplied baseline configuration differs from the baseline recipe.")
            print(f"Reusing archived {model_name} baseline: {source}")
        else:
            source = candidate_dir
            if (source / "training_summary.json").is_file():
                old_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
                if old_config != asdict(config):
                    raise ValueError(f"Changed recipe in {source}. Choose a new RUN_NAME.")
                verify_split(source / "split_manifest.json", data)
                old_data = json.loads((source / "data_config.json").read_text(encoding="utf-8"))
                if old_data != data.settings:
                    raise ValueError(f"Changed loader settings in {source}. Choose a new RUN_NAME.")
                print(f"Reusing completed experiment: {source}")
            elif (source / "best.pt").exists():
                raise ValueError(f"Incomplete run in {source}. Preserve it and choose a new RUN_NAME; optimizer resume is unavailable.")
            else:
                print(f"\n=== {model_name}: {name} ===")
                fit(config, data, source)
        # Re-evaluate every candidate identically, even archived AMP-era baselines.
        metrics = validate_checkpoint(source / "best.pt", data, candidate_dir, device=config.device)
        rows.append({"candidate": name, "val_accuracy": metrics["accuracy"],
                     "val_macro_f1": metrics["macro_f1"], "val_loss": metrics["loss"],
                     "epoch": metrics["epoch"], "augmentation": config.augmentation,
                     "checkpoint": str(source / "best.pt")})
        sources[name] = source
        print(f"{name}: validation accuracy {metrics['accuracy']:.2%}, F1 {metrics['macro_f1']:.4f}")
    winner = select_candidate(rows)
    selected_dir = output_root / model_name
    selected_dir.mkdir(parents=True, exist_ok=True)
    # Copy only experiment provenance/training artifacts; never import old test scores.
    filenames = ["best.pt", "last.pt", "config.json", "data_config.json", "training_summary.json",
                 "class_counts.json", "split_manifest.json", "history.csv", "training_curves.png"]
    for filename in filenames:
        source = sources[winner["candidate"]] / filename
        if source.is_file():
            shutil.copy2(source, selected_dir / filename)
    for path in (experiment_root / winner["candidate"]).glob("validation_*"):
        shutil.copy2(path, selected_dir / path.name)
    baseline = next(row for row in rows if row["candidate"] == "baseline")
    result = {"model": model_name, "selected_candidate": winner["candidate"],
              "criterion": "validation_accuracy_then_macro_f1", "candidates": rows,
              "validation_accuracy_gain": winner["val_accuracy"] - baseline["val_accuracy"],
              "checkpoint": str(selected_dir / "best.pt")}
    _write_json(result, selected_dir / "selection.json")
    with (experiment_root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {model_name}/{winner['candidate']} by validation accuracy. "
          f"Gain over baseline: {result['validation_accuracy_gain']:+.2%}")
    return result
