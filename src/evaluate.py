"""Generate the lab's metrics, training curves, confusion matrices and errors."""

import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import CLASS_NAMES
from src.model import count_parameters
from src.predict import load_checkpoint
from src.train import resolve_device


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def plot_class_distribution(data, output_path):
    counts = data.class_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(7)
    for offset, split in zip([-0.25, 0, 0.25], ["train", "val", "test"]):
        ax.bar(x + offset, list(counts[split].values()), width=0.25, label=split)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("Images")
    ax.set_title("FER-2013 class distribution (validation is held out of train)")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


def plot_history(history_path, output_path=None):
    history_path = Path(history_path)
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ["train", "val"]:
        axes[0].plot(epochs, [float(r[f"{split}_loss"]) for r in rows], label=split)
        axes[1].plot(epochs, [float(r[f"{split}_accuracy"]) for r in rows], label=split)
    axes[0].set_title("Plain cross entropy (train includes augmentation/dropout)")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output_path = Path(output_path or history_path.with_name("training_curves.png"))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def benchmark_cpu(model, warmup=10, iterations=100, num_threads=4):
    """Single-face, batch-one forward passes; excludes detection and preprocessing."""
    if iterations < 1 or warmup < 0 or num_threads < 1:
        raise ValueError("Invalid benchmark settings.")
    old_threads = torch.get_num_threads()
    old_device = next(model.parameters()).device
    old_training = model.training
    try:
        torch.set_num_threads(num_threads)
        model.to("cpu").eval()
        example = torch.zeros(1, 1, 48, 48)
        with torch.inference_mode():
            for _ in range(warmup):
                model(example)
            start = time.perf_counter()
            for _ in range(iterations):
                model(example)
            elapsed = time.perf_counter() - start
        return {"cpu_model_fps": iterations / elapsed, "cpu_model_ms": elapsed * 1000 / iterations,
                "benchmark_iterations": iterations, "cpu_threads": num_threads,
                "cpu": platform.processor() or platform.machine(),
                "benchmark_scope": "CPU model forward only; batch=1, 1x48x48, float32; excludes face detection, I/O and preprocessing"}
    finally:
        model.to(old_device).train(old_training)
        torch.set_num_threads(old_threads)


def plot_confusion(matrix, output_path, title):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    threshold = matrix.max() / 2
    for row in range(7):
        for col in range(7):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    ax.set_xticks(range(7), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(7), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def evaluate_checkpoint(checkpoint_path, data, output_dir=None, device="auto", benchmark_iterations=100):
    """Evaluate the selected best checkpoint on test once the recipe is fixed."""
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    output_dir = Path(output_dir or Path(checkpoint_path).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets, predictions, probabilities = [], [], []
    for images, labels in data.test_loader:
        scores = model(images.to(device)).softmax(dim=1).cpu()
        targets.extend(labels.tolist())
        predictions.extend(scores.argmax(dim=1).tolist())
        probabilities.extend(scores.tolist())
    targets, predictions, probabilities = np.array(targets), np.array(predictions), np.array(probabilities)
    accuracy = float(accuracy_score(targets, predictions))
    target_accuracy = 0.60 if checkpoint["model_name"] == "cnn" else 0.65
    metrics = {
        "model": checkpoint["model_name"], "test_accuracy": accuracy,
        "test_macro_f1": float(f1_score(targets, predictions, labels=range(7), average="macro", zero_division=0)),
        "parameters": count_parameters(model), "test_images": len(targets),
        "best_epoch": checkpoint["epoch"], "target_accuracy": target_accuracy,
        "target_met": accuracy >= target_accuracy,
        **benchmark_cpu(model, iterations=benchmark_iterations),
    }
    save_json(metrics, output_dir / "metrics.json")
    report = classification_report(targets, predictions, labels=list(range(7)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    save_json(report, output_dir / "classification_report.json")
    matrix = confusion_matrix(targets, predictions, labels=list(range(7)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion(matrix, output_dir / "confusion_matrix.png", f"{checkpoint['model_name']} - test confusion matrix")
    wrong_indices = np.flatnonzero(targets != predictions)
    # Choose errors across true classes before filling the remaining slots.
    selected = []
    for label in range(7):
        candidates = [int(i) for i in wrong_indices if targets[i] == label]
        selected.extend(candidates[:1])
    selected.extend(int(i) for i in wrong_indices if int(i) not in selected)
    selected = selected[:10]
    example_numbers = {index: number for number, index in enumerate(selected, start=1)}
    rows = []
    selected_set = set(selected)
    for index, (path, label) in enumerate(data.test.samples):
        predicted = int(predictions[index])
        confidence = float(probabilities[index, predicted])
        true_probability = float(probabilities[index, label])
        comment = ""
        if index in selected_set:
            gap = confidence - true_probability
            comment = (f"Assigned {CLASS_NAMES[predicted]} {confidence:.1%}, versus "
                       f"{CLASS_NAMES[label]} {true_probability:.1%} (gap {gap:.1%}). "
                       + ("Confident error; inspect label ambiguity, face crop and domain mismatch."
                          if confidence >= 0.8 else
                          "Inspect the shared facial cues and whether this label is visually ambiguous."))
        rows.append({"index": index, "path": str(Path(path).relative_to(data.root)),
                     "example_number": example_numbers.get(index, ""),
                     "true": CLASS_NAMES[label], "predicted": CLASS_NAMES[predicted],
                     "confidence": confidence, "true_probability": true_probability,
                     "correct": bool(label == predicted), "comment": comment})
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if selected:
        fig, axes = plt.subplots(2, 5, figsize=(13, 6))
        for axis in axes.flat:
            axis.axis("off")
        for axis, index in zip(axes.flat, selected):
            with Image.open(data.test.samples[index][0]) as image:
                axis.imshow(image.convert("L"), cmap="gray")
            row = rows[index]
            axis.set_title(f"#{row['example_number']} True: {row['true']}\n"
                           f"Pred: {row['predicted']} ({row['confidence']:.0%})", fontsize=9)
        fig.suptitle(f"{checkpoint['model_name']} - {len(selected)} test errors")
        fig.tight_layout()
        fig.savefig(output_dir / "misclassified.png", dpi=150)
        plt.close(fig)
    else:
        (output_dir / "misclassified.png").unlink(missing_ok=True)
    notes = [f"# {checkpoint['model_name']} error analysis", "",
             f"Selected {len(selected)} errors out of {len(wrong_indices)} test mistakes.",
             "These are probability-based observations and inspection prompts; add your own visual observations.", ""]
    for index in selected:
        notes.extend([f"- #{rows[index]['example_number']} `{rows[index]['path']}`: {rows[index]['comment']}"])
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    notes.extend(["", "## Most frequent confusions", ""])
    for flat_index in np.argsort(off_diagonal.ravel())[::-1][:5]:
        true, pred = np.unravel_index(flat_index, (7, 7))
        if off_diagonal[true, pred] > 0:
            notes.append(f"- {CLASS_NAMES[true]} -> {CLASS_NAMES[pred]}: {off_diagonal[true, pred]} images.")
    (output_dir / "error_analysis.md").write_text("\n".join(notes), encoding="utf-8")
    if (output_dir / "history.csv").exists():
        plot_history(output_dir / "history.csv")
    return metrics


def compare_models(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "test_accuracy", "test_macro_f1", "parameters", "cpu_model_fps",
              "cpu_model_ms", "target_accuracy", "target_met"]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    save_json(metrics, output_dir / "comparison.json")
    return [{key: result[key] for key in fields} for result in metrics]
