"""M1 matches the supplied architecture; M2 fine-tunes AffectNet-pretrained DDAMFN."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )


class EmotionCNN(nn.Module):
    """48x48 -> 24 -> 12 -> 6 -> 3; flatten 3*3*256 = 2304."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, 0.10),
            ConvBlock(32, 64, 0.15),
            ConvBlock(64, 128, 0.20),
            ConvBlock(128, 256, 0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss includes log-softmax: return raw scores during training.
        return self.classifier(self.features(images))


def build_resnet18(pretrained: bool = True, num_classes: int = 7) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    original_conv = model.conv1
    # Keep 48x48 detail: use stride 1 and remove the initial max-pooling step.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
    if pretrained:
        # For repeated grayscale input, sum_c W_c * gray == (sum_c W_c) * gray.
        # This preserves the RGB stem's response to three identical normalized channels.
        with torch.no_grad():
            model.conv1.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))
    else:
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(nn.Dropout(0.30), nn.Linear(model.fc.in_features, num_classes))
    return model


def model_architecture(name, architecture=None):
    if name not in {"cnn", "transfer"}:
        raise ValueError("Choose model 'cnn' or 'transfer'.")
    architecture = architecture or ("cnn" if name == "cnn" else "ddamfn")
    if architecture not in ({"cnn"} if name == "cnn" else {"ddamfn", "resnet18"}):
        raise ValueError(f"Unsupported architecture {architecture!r} for {name}.")
    return architecture


def build_model(name: str, pretrained: bool = False, architecture=None, pretrained_path=None) -> nn.Module:
    architecture = model_architecture(name, architecture)
    if name == "cnn":
        return EmotionCNN()
    if architecture == "ddamfn":
        from src.ddamfn import DDAMFN, load_affectnet_weights
        model = DDAMFN()
        return load_affectnet_weights(model, pretrained_path) if pretrained else model
    if architecture == "resnet18":
        return build_resnet18(pretrained=pretrained)
    raise ValueError(f"Unknown model {name!r}; choose 'cnn' or 'transfer'.")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
