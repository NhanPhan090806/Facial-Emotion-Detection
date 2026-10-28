"""Load a saved checkpoint and share exactly the same preprocessing as training."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data import CLASS_NAMES, make_transform, preprocessing_for
from src.model import build_model
from src.train import resolve_device


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train first or copy the selected checkpoints into models/.")
    device = resolve_device(str(device))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match this project.")
    # Version-1 transfer checkpoints are the previous ResNet-18 experiments.
    architecture = checkpoint.get("architecture", "cnn" if checkpoint["model_name"] == "cnn" else "resnet18")
    expected = preprocessing_for(architecture)
    if checkpoint.get("preprocessing") != expected:
        raise ValueError("Checkpoint preprocessing does not match its architecture.")
    checkpoint["architecture"] = architecture
    # Loading saved weights must never trigger a pretrained download.
    model = build_model(checkpoint["model_name"], pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


class EmotionPredictor:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.transform = make_transform(training=False, architecture=self.checkpoint["architecture"])
        self.class_names = self.checkpoint["class_names"]

    @torch.inference_mode()
    def predict_faces(self, faces):
        """faces is a sequence of OpenCV BGR arrays (or uint8 grayscale arrays)."""
        if len(faces) == 0:
            return []
        images = []
        for face in faces:
            if not isinstance(face, np.ndarray) or face.size == 0:
                raise ValueError("Each face must be a non-empty NumPy image.")
            if face.ndim == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            elif face.ndim != 2:
                raise ValueError("Expected a grayscale or BGR face image.")
            images.append(self.transform(Image.fromarray(face)))
        batch = torch.stack(images).to(self.device)
        probabilities = self.model(batch).softmax(dim=1).cpu()
        results = []
        for probability in probabilities:
            confidence, index = probability.max(dim=0)
            results.append({"label": self.class_names[index.item()],
                            "confidence": confidence.item(), "probabilities": probability.tolist()})
        return results

    def predict(self, face):
        return self.predict_faces([face])[0]
