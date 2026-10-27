"""Small, accuracy-first validation search. Test images never choose the winner."""

import csv
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import CLASS_NAMES, build_data, preprocessing_for
from src.predict import load_checkpoint
from src.model import model_architecture
from src.train import TrainConfig, fit, resolve_device


def accuracy_recipes(model_name, seed=42, device="auto", stage=3):
    """Compare a control with new candidates; stage=2 reproduces the previous search.

    In stage 3, M2's baseline is the previous lighting winner. New candidates
    isolate erasing at the original LR and extend the slower regularized recipe.
    """
    if model_name not in {"cnn", "transfer"}:
        raise ValueError("Choose cnn or transfer.")
    baseline = TrainConfig(model_name=model_name, architecture="cnn" if model_name == "cnn" else "resnet18", epochs=80 if model_name == "cnn" else 20,
                           pretrained=model_name == "transfer", seed=seed, device=device,
                           patience=18 if model_name == "cnn" else 12)
    if stage == 2:
        recipes = {"baseline": baseline, "lighting": replace(baseline, augmentation="lighting")}
        if model_name == "transfer":
            recipes["regularized"] = replace(
                baseline, augmentation="occlusion", backbone_lr=5e-5, weight_decay=5e-4,
            )
        return recipes
    if stage != 3:
        raise ValueError("stage must be 2 or 3.")
    if model_name == "cnn":
        return {"baseline": baseline,
                "gentle_combined": replace(baseline, augmentation="gentle_combined")}
    control = replace(baseline, augmentation="lighting")
    return {
        "baseline": control,
        "lighting_erasing": replace(control, augmentation="occlusion"),
        "regularized_long": replace(control, augmentation="occlusion", epochs=40,
                                    backbone_lr=5e-5, weight_decay=5e-4),
    }


def _write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _normalized_split(rows):
    # Compare Kaggle/Linux and Windows manifests using the same path separators.
    return sorted((row["path"].replace("\\", "/"), int(row["label"])) for row in rows)


def read_archive(root, model_name):
    """Accept full outputs/model/ or compact outputs/ + sibling weights/ archives."""
    root = Path(root)
    source = root / model_name
    checkpoint_path = source / "best.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = root.parent / "weights" / f"{model_name}_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No {model_name} checkpoint under {root} or its sibling weights/.")
    manifest_path = source / "split_manifest.json"
    if not manifest_path.is_file():
        manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Archived model needs its original split manifest: {root}")
    metadata = None
    if (source / "config.json").is_file() and (source / "best.pt").is_file():
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    else:
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if metadata.get("model_name") != model_name:
            raise ValueError("Archived checkpoint model name does not match its requested model.")
        config = metadata["config"]
    config.setdefault("architecture", "cnn" if model_name == "cnn" else "resnet18")
    return {"source": source, "checkpoint": checkpoint_path, "manifest": manifest_path,
            "config": config, "metadata": metadata}


def materialize_compact_archive(archive, data, destination):
    """Stage exact weights and recorded metadata; never invent an epoch history."""
    destination = Path(destination)
    metadata = archive["metadata"]
    shutil.copy2(archive["checkpoint"], destination / "best.pt")
    _write_json(archive["config"], destination / "config.json")
    _write_json(data.class_counts(), destination / "class_counts.json")
    data.save_manifest(destination / "split_manifest.json")
    original_val = metadata["val_metrics"]
    summary = {"best_epoch": metadata["epoch"], "best_val_accuracy": original_val["accuracy"],
               "best_val_macro_f1": original_val["macro_f1"], "training_seconds": None,
               "device": "archived (not recorded)", "history_available": False,
               "source_checkpoint": str(archive["checkpoint"])}
    _write_json(summary, destination / "training_summary.json")
    print("Archived weights have no epoch history; retain the original report for training plots.")
    return destination


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
    if data.settings.get("preprocessing", preprocessing_for()) != checkpoint["preprocessing"]:
        raise ValueError("Validation loaders must use the checkpoint's preprocessing architecture.")
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
                             recipes=None, reference_results_roots=None):
    """Train/cache candidates, rank ONLY on validation, export one selected model.

    Optional baseline_results_root points to an old outputs/ with model/best.pt
    and model/split_manifest.json. For stage 3 use run 2's selected outputs/.
    reference_results_roots optionally maps names (e.g. run1) to other old outputs/
    folders. They enter validation selection without retraining or reading test scores.
    Missing or mismatched baselines fail explicitly.
    Completed matching experiments are reused; interrupted fits cannot resume.
    """
    recipes = accuracy_recipes(model_name, seed, device) if recipes is None else recipes
    recipes = {name: replace(config, architecture=model_architecture(model_name, config.architecture))
               for name, config in recipes.items()}
    if "baseline" not in recipes:
        raise ValueError("Include baseline in the candidate recipes.")
    output_root = Path(output_root)
    experiment_root = output_root / "experiments" / model_name
    rows, sources = [], {}
    jobs = {name: (config, None) for name, config in recipes.items()}
    for label, root in (reference_results_roots or {}).items():
        name = f"reference_{label}"
        if name in jobs:
            raise ValueError(f"Duplicate candidate name: {name}")
        archive = read_archive(root, model_name)
        config = replace(TrainConfig(**archive["config"]), device=device)
        jobs[name] = (config, archive)
    for name, (config, reference_source) in jobs.items():
        if config.model_name != model_name or name in {"", ".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Invalid candidate name or model configuration.")
        data = build_data(data_root, batch_size=batch_size, num_workers=num_workers,
                          val_fraction=val_fraction, seed=split_seed, augmentation=config.augmentation,
                          architecture=model_architecture(model_name, config.architecture))
        candidate_dir = experiment_root / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        reused_baseline = name == "baseline" and baseline_results_root is not None
        if reused_baseline or reference_source is not None:
            archive = reference_source if reference_source is not None else read_archive(baseline_results_root, model_name)
            verify_split(archive["manifest"], data)
            old_config = archive["config"]
            old_config = {**asdict(TrainConfig()), **old_config}
            # The first run predates the augmentation key; that means baseline.
            expected = asdict(config)
            ignored = {"device", "cpu_threads", "amp"}
            if any(old_config.get(key, "baseline" if key == "augmentation" else None) != value
                   for key, value in expected.items() if key not in ignored):
                raise ValueError("The archived configuration differs from the candidate recipe. Stage 3's control expects run 2's selected outputs/.")
            source = archive["source"]
            if archive["metadata"] is not None:
                source = materialize_compact_archive(archive, data, candidate_dir)
            print(f"Reusing archived {model_name}/{name}: {source}")
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
