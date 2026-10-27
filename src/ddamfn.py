"""DDAMFN with the official AffectNet-7 DDAMFN++ initialization.

Architecture reference: https://github.com/SainingZhang/DDAMFN
This implementation keeps upstream parameter names for strict state-dict loading.
Only tensor state dictionaries are loaded; inference never needs upstream code.
"""

import hashlib
from pathlib import Path
import urllib.request

import torch
from torch import nn

UPSTREAM_COMMIT = "cebfe5c5969b7d1fd64ecd775e318d3834ca76bb"
PRETRAINED_URL = (
    "https://raw.githubusercontent.com/SainingZhang/DDAMFN/" + UPSTREAM_COMMIT
    + "/DDAMFN%2B%2B/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth"
)
PRETRAINED_SHA256 = "edf731a458162eef6b2be12c3572bd10cbe92ce8c009da4331184a2943d7ed14"
# AffectNet: neutral, happy, sad, surprise, fear, disgust, angry.
# Project:   angry, disgust, fear, happy, neutral, sad, surprise.
CLASS_ROW_ORDER = [6, 5, 4, 1, 0, 2, 3]


class ConvBN(nn.Module):
    def __init__(self, inputs, outputs, kernel=1, stride=1, padding=0, groups=1, activation=False):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(outputs)
        self.prelu = nn.PReLU(outputs) if activation else nn.Identity()

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class CoordinateAttention(nn.Module):
    """Combine a vertical and a horizontal attention map for each channel."""
    def __init__(self, channels, learned_pool=False):
        super().__init__()
        self.learned_pool = learned_pool
        if learned_pool:
            self.Linear_h = ConvBN(channels, channels, (1, 7), groups=channels)
            self.Linear_w = ConvBN(channels, channels, (7, 1), groups=channels)
            # Present but unused in the official forward; retain for checkpoint compatibility.
            self.Linear = ConvBN(channels, channels, 7, groups=channels)
        else:
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden = max(8, channels // 32)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        vertical = self.Linear_h(x) if self.learned_pool else self.pool_h(x)
        horizontal = self.Linear_w(x) if self.learned_pool else self.pool_w(x)
        combined = torch.cat([vertical, horizontal.transpose(2, 3)], dim=2)
        combined = nn.functional.hardswish(self.bn1(self.conv1(combined)))
        vertical, horizontal = combined.split([height, width], dim=2)
        a_h = self.conv2(vertical).sigmoid()
        a_w = self.conv3(horizontal.transpose(2, 3)).sigmoid()
        attention = a_w * a_h
        return attention if self.learned_pool else x * attention


class MixedDepthwise(nn.Module):
    """Apply different kernel sizes to separate groups of channels."""
    def __init__(self, splits, kernels, stride):
        super().__init__()
        self.splits = splits
        self.mixed_depthwise_conv = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel, stride, kernel // 2,
                      groups=channels, bias=False)
            for channels, kernel in zip(splits, kernels)
        ])
        # These layers are also unused in the upstream forward, but stored in its weights.
        self.bn = nn.BatchNorm2d(sum(splits))
        self.prelu = nn.PReLU(sum(splits))

    def forward(self, x):
        parts = x.split(self.splits, dim=1)
        return torch.cat([conv(part) for conv, part in zip(self.mixed_depthwise_conv, parts)], dim=1)


class MixedBlock(nn.Module):
    def __init__(self, inputs, outputs, splits, kernels, stride=1, residual=False):
        super().__init__()
        expanded = sum(splits)
        self.conv = ConvBN(inputs, expanded, activation=True)
        self.conv_dw = MixedDepthwise(splits, kernels, stride)
        self.CA = CoordinateAttention(expanded)
        self.project = ConvBN(expanded, outputs)
        self.residual = residual

    def forward(self, x):
        features = self.project(self.CA(self.conv_dw(self.conv(x))))
        return x + features if self.residual else features


class MixedStage(nn.Module):
    def __init__(self, channels, blocks, splits, kernels):
        super().__init__()
        self.model = nn.Sequential(*[
            MixedBlock(channels, channels, splits, kernels, residual=True)
            for _ in range(blocks)
        ])

    def forward(self, x):
        return self.model(x)


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordinateAttention(512, learned_pool=True)

    def forward(self, x):
        return self.CoordAtt(x)


class DDAMFN(nn.Module):
    """112x112 repeated grayscale -> face features -> two attention heads -> 7 logits."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(3, 64, 3, stride=2, padding=1, activation=True),
            ConvBN(64, 64, 3, padding=1, groups=64, activation=True),
            MixedBlock(64, 64, [64, 32, 32], [3, 5, 7], stride=2),
            MixedStage(64, 9, [96, 32], [3, 5]),
            MixedBlock(64, 128, [128, 64, 64], [3, 5, 7], stride=2),
            MixedStage(128, 16, [192, 64], [3, 5]),
            MixedBlock(128, 256, [256, 256, 256, 256], [3, 5, 7, 9], stride=2),
            MixedStage(256, 6, [172, 170, 170], [3, 5, 7]),
            ConvBN(256, 512, activation=True),
        )
        self.cat_head0 = AttentionHead()
        self.cat_head1 = AttentionHead()
        self.Linear = ConvBN(512, 512, 7, groups=512)
        self.fc = nn.Linear(512, 7)
        self.bn = nn.BatchNorm1d(7)  # Unused upstream, kept for strict weight compatibility.
        self.pretrained_source = None

    def forward_with_attention(self, images):
        if images.shape[1:] != (3, 112, 112):
            raise ValueError("DDAMFN expects N x 3 x 112 x 112 input.")
        features = self.features(images)
        heads = [self.cat_head0(features), self.cat_head1(features)]
        attended = features * torch.maximum(heads[0], heads[1])
        logits = self.fc(self.Linear(attended).flatten(1))
        return logits, heads

    def forward(self, images):
        return self.forward_with_attention(images)[0]


def download_pretrained(path=None):
    """Download once with a pinned revision and full SHA-256 integrity check."""
    path = Path(path) if path else Path(torch.hub.get_dir()) / "checkpoints" / "ddamfn_affectnet7.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(".download")
        print("Downloading the official AffectNet-7 DDAMFN++ checkpoint (about 17 MB)...", flush=True)
        try:
            urllib.request.urlretrieve(PRETRAINED_URL, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != PRETRAINED_SHA256:
                raise ValueError("DDAMFN download checksum mismatch; refusing unverified weights.")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRETRAINED_SHA256:
        raise ValueError(f"DDAMFN checkpoint checksum mismatch: {path}")
    return path


def load_affectnet_weights(model, path=None):
    path = download_pretrained(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    # Reuse emotion knowledge, remapping classifier rows to the project's class order.
    for key in ["fc.weight", "fc.bias", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]:
        state[key] = state[key][CLASS_ROW_ORDER]
    model.load_state_dict(state, strict=True)
    model.pretrained_source = {
        "dataset": "AffectNet-7", "variant": "DDAMFN++ checkpoint / DDAMFN architecture",
        "url": PRETRAINED_URL, "commit": UPSTREAM_COMMIT,
        "sha256": PRETRAINED_SHA256, "class_row_order": CLASS_ROW_ORDER,
    }
    return model
