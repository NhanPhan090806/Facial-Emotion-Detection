"""Explicit PyTorch training loop shared by the local and Kaggle notebooks."""

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.data import CLASS_NAMES, DataBundle, preprocessing_for
from src.model import build_model, count_parameters, model_architecture


@dataclass
class TrainConfig:
    model_name: str = "cnn"
    epochs: int = 80
    learning_rate: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 18
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    pretrained: bool = False
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    cpu_threads: int = 4
    augmentation: str = "baseline"
    architecture: str | None = None
    pretrained_path: str | None = None
    warmup_epochs: int = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Choose device='cpu'.")
    return device


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, total_objective, total_examples = 0.0, 0.0, 0
    targets, predictions = [], []
    optimizer_steps, skipped_steps = 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss. Check the data and learning rate.")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped_steps += 1  # Non-finite scaled gradients: GradScaler skipped this update.
                else:
                    optimizer_steps += 1
            total_objective += loss.item() * labels.size(0)
            # Comparable, unsmoothed/unweighted losses for the train/val plot.
            plain_loss = nn.functional.cross_entropy(logits.detach().float(), labels)
            total_loss += plain_loss.item() * labels.size(0)
            total_examples += labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if not total_examples:
        raise ValueError("Cannot train or evaluate an empty loader.")
    result = {
        "loss": total_loss / total_examples,
        "objective_loss": total_objective / total_examples,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(7)),
                                    average="macro", zero_division=0)),
    }
    if training:
        result.update(optimizer_steps=optimizer_steps, skipped_steps=skipped_steps)
    return result


def fit(config: TrainConfig, data: DataBundle, output_dir):
    """Select checkpoints ONLY on validation accuracy, with macro-F1 as a tie-breaker.

    A fresh run overwrites files in output_dir; choose a new directory per experiment.
    Test data is never accessed by this function.
    """
    if config.epochs < 1 or config.patience < 1 or config.cpu_threads < 1:
        raise ValueError("epochs, patience and cpu_threads must be positive.")
    if config.learning_rate <= 0 or config.backbone_lr <= 0:
        raise ValueError("Learning rates must be positive.")
    if not 0 <= config.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1).")
    if config.augmentation != data.settings.get("augmentation", "baseline"):
        raise ValueError("Build the data loaders using the config's augmentation profile.")
    architecture = model_architecture(config.model_name, config.architecture)
    if data.settings.get("architecture", "cnn") != architecture:
        raise ValueError("Data preprocessing architecture must match the model architecture.")
    if config.warmup_epochs < 0 or config.warmup_epochs >= config.epochs:
        raise ValueError("warmup_epochs must be nonnegative and less than epochs.")
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config.model_name, pretrained=config.pretrained, architecture=architecture,
                        pretrained_path=config.pretrained_path).to(device)
    if config.model_name == "transfer":
        head = list(model.fc.parameters())
        backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
        parameters = [{"params": backbone, "lr": config.backbone_lr},
                      {"params": head, "lr": config.learning_rate}]
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    amp = config.amp and device.type == "cuda"
    # A conservative initial scale reduces skipped startup updates in the pretrained face model.
    scaler = torch.amp.GradScaler("cuda", enabled=amp,
                                 init_scale=1024.0 if architecture == "ddamfn" else 65536.0)
    weights = data.class_weights.to(device) if config.use_class_weights else None
    # reduction='none' then mean makes per-epoch weighted-loss aggregation exact.
    # Log losses use plain cross entropy on BOTH splits for interpretable curves.
    weighted_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing,
                                       reduction="none")
    def criterion(scores, labels):
        return weighted_loss(scores, labels).mean()

    eval_criterion = nn.CrossEntropyLoss()
    best_score, best_epoch, stale = (-1.0, -1.0), 0, 0
    history = []
    config_dict = {**asdict(config), "architecture": architecture}
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    (output_dir / "data_config.json").write_text(json.dumps(data.settings, indent=2), encoding="utf-8")
    data.save_manifest(output_dir / "split_manifest.json")
    (output_dir / "class_counts.json").write_text(json.dumps(data.class_counts(), indent=2), encoding="utf-8")
    (output_dir / "pretrained_source.json").write_text(
        json.dumps(getattr(model, "pretrained_source", None), indent=2), encoding="utf-8",
    )
    print(f"{config.model_name}: {count_parameters(model):,} parameters | {device} | AMP={amp}", flush=True)
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        # Epoch-based cosine schedule: skipped AMP updates do not cause scheduler warnings.
        cosine = (1 + np.cos(np.pi * (epoch - 1) / config.epochs)) / 2
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * cosine
        if config.model_name == "transfer" and config.warmup_epochs:
            optimizer.param_groups[0]["lr"] *= min(1.0, epoch / config.warmup_epochs)
        train_metrics = run_epoch(model, data.train_loader, criterion, device, optimizer, scaler, amp)
        # Select in float32, matching deployment inference rather than training autocast.
        val_metrics = run_epoch(model, data.val_loader, eval_criterion, device, amp=False)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[-1]["lr"],
               "backbone_learning_rate": optimizer.param_groups[0]["lr"],
               **{f"train_{key}": value for key, value in train_metrics.items()},
               **{f"val_{key}": value for key, value in val_metrics.items()},
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        score = (val_metrics["accuracy"], val_metrics["macro_f1"])
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        checkpoint = {
            "format_version": 2, "model_name": config.model_name, "architecture": architecture,
            "model_state": model.state_dict(), "class_names": CLASS_NAMES,
            "preprocessing": preprocessing_for(architecture),
            "pretrained_source": getattr(model, "pretrained_source", None),
            "epoch": epoch, "best_epoch": best_epoch, "val_metrics": val_metrics,
            "config": config_dict, "parameters": count_parameters(model),
            "data_config": data.settings,
        }
        # Portable inference checkpoints: no optimizer objects or arbitrary Python objects.
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:03d}/{config.epochs:03d} | train loss {row['train_loss']:.4f} "
              f"acc {row['train_accuracy']:.3f} | val loss {row['val_loss']:.4f} "
              f"acc {row['val_accuracy']:.3f} F1 {row['val_macro_f1']:.3f} | {row['seconds']:.1f}s" +
              (f" | AMP skipped {train_metrics['skipped_steps']} updates"
               if train_metrics["skipped_steps"] else "") + (" *" if improved else ""), flush=True)
        if stale >= config.patience:
            print(f"Early stopping; best validation epoch: {best_epoch}.")
            break
    summary = {"best_epoch": best_epoch, "best_val_accuracy": best_score[0],
               "best_val_macro_f1": best_score[1], "training_seconds": time.perf_counter() - start,
               "device": str(device), "torch_version": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"checkpoint": output_dir / "best.pt", "history": history, **summary}
