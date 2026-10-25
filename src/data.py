"""FER-2013 ImageFolder loading with a reproducible train/validation split.

The Kaggle dataset already has train/ and test/. Validation comes ONLY from train/.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# ImageFolder's alphabetical order, also stored in every checkpoint.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = 48
MEAN = (0.5,)
STD = (0.5,)


def make_transform(training: bool = False, augmentation: str = "baseline"):
    """Lighting/occlusion are training-only; inference remains exactly unchanged."""
    if augmentation not in {"baseline", "lighting", "occlusion"}:
        raise ValueError("augmentation must be baseline, lighting, or occlusion.")
    steps = [transforms.Grayscale(num_output_channels=1), transforms.Resize((48, 48))]
    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10, translate=(0.08, 0.08), scale=(0.90, 1.10),
                interpolation=InterpolationMode.BILINEAR, fill=128,
            ),
        ])
        if augmentation in {"lighting", "occlusion"}:
            steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if training and augmentation == "occlusion":
        # Zero is mid-gray after normalization. Keep the erased region small at 48x48.
        steps.append(transforms.RandomErasing(
            p=0.10, scale=(0.02, 0.06), ratio=(0.5, 2.0), value=0.0,
        ))
    return transforms.Compose(steps)


def find_data_root(path) -> Path:
    """Accept the FER root or a parent, including Kaggle's nested input folders."""
    path = Path(path).expanduser().resolve()
    if (path / "train").is_dir() and (path / "test").is_dir():
        return path
    candidates = sorted({folder.parent for folder in path.rglob("train")
                         if folder.is_dir() and (folder.parent / "test").is_dir()})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No FER-2013 train/ and test/ folders under {path}.")
    raise ValueError(f"Multiple datasets found under {path}; specify one: {candidates}")


class FaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("L"))
        return image, label


@dataclass
class DataBundle:
    root: Path
    train: FaceDataset
    val: FaceDataset
    test: FaceDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: torch.Tensor
    settings: dict = field(default_factory=dict)

    def class_counts(self):
        return {split: dict(zip(CLASS_NAMES, np.bincount(dataset.targets, minlength=7).tolist()))
                for split, dataset in [("train", self.train), ("val", self.val), ("test", self.test)]}

    def save_manifest(self, path):
        manifest = {split: [{"path": str(Path(p).relative_to(self.root)), "label": label}
                            for p, label in dataset.samples]
                    for split, dataset in [("train", self.train), ("val", self.val), ("test", self.test)]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_data(data_root, batch_size=128, num_workers=0, val_fraction=0.1, seed=42,
               augmentation="baseline"):
    if batch_size < 2:
        raise ValueError("Use batch_size >= 2 because M1 has BatchNorm1d.")
    if num_workers < 0 or not 0 < val_fraction < 1:
        raise ValueError("num_workers must be >= 0 and val_fraction must be between 0 and 1.")
    root = find_data_root(data_root)
    train_folder = datasets.ImageFolder(root / "train")
    test_folder = datasets.ImageFolder(root / "test")
    for dataset in [train_folder, test_folder]:
        if dataset.classes != CLASS_NAMES:
            raise ValueError(f"Expected class folders {CLASS_NAMES}; found {dataset.classes}.")
    train_indices, val_indices = train_test_split(
        np.arange(len(train_folder)), test_size=val_fraction,
        random_state=seed, stratify=train_folder.targets,
    )
    train = FaceDataset([train_folder.samples[i] for i in train_indices], make_transform(True, augmentation))
    val = FaceDataset([train_folder.samples[i] for i in val_indices], make_transform(False))
    test = FaceDataset(test_folder.samples, make_transform(False))
    counts = np.bincount(train.targets, minlength=7)
    if np.any(counts == 0):
        raise ValueError("Every emotion must occur in the training split.")
    # Square-root inverse frequency tempers very large weights for rare 'disgust'.
    weights = np.sqrt(counts.sum() / (7 * counts))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    options = dict(batch_size=batch_size, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(seed)
    # Avoid a singleton final batch (the dense BatchNorm needs >1 training example).
    drop_last = len(train) % batch_size == 1
    train_loader = DataLoader(train, shuffle=True, generator=generator, drop_last=drop_last, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    test_loader = DataLoader(test, shuffle=False, **options)
    settings = {"augmentation": augmentation, "batch_size": batch_size,
                "num_workers": num_workers, "split_seed": seed, "val_fraction": val_fraction,
                "class_weight_rule": "sqrt_inverse_frequency_normalized_mean_one",
                "class_weights": weights.tolist()}
    return DataBundle(root, train, val, test, train_loader, val_loader, test_loader, weights, settings)
