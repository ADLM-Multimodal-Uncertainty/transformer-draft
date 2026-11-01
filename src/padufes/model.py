"""Simple late-fusion network, also supporting image-only and metadata-only ablations."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 6,
        mode: str = "multimodal",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if mode not in {"multimodal", "image", "metadata"}:
            raise ValueError("mode must be 'multimodal', 'image', or 'metadata'")
        if mode != "metadata" and freeze_backbone and not pretrained:
            raise ValueError("Cannot freeze a randomly initialized backbone")
        self.mode = mode
        self.freeze_backbone = freeze_backbone

        if mode in {"multimodal", "image"}:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.image_backbone = resnet18(weights=weights)
            self.image_backbone.fc = nn.Identity()  # ResNet18 -> 512 dimensions
            if freeze_backbone:
                for param in self.image_backbone.parameters():
                    param.requires_grad_(False)
            self.image_head = nn.Sequential(
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2)
            )
        else:
            self.image_backbone = None
            self.image_head = None

        if mode in {"multimodal", "metadata"}:
            if metadata_dim < 1:
                raise ValueError("metadata_dim must be > 0")
            self.metadata_head = nn.Sequential(
                nn.Linear(metadata_dim, 64), nn.ReLU(), nn.Dropout(0.2)
            )
        else:
            self.metadata_head = None

        dim = {"multimodal": 192, "image": 128, "metadata": 64}[mode]
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # With a frozen pretrained backbone, also freeze BatchNorm statistics.
        if mode and self.image_backbone is not None and self.freeze_backbone:
            self.image_backbone.eval()
        return self

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        features = []
        if self.image_backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    image_embedding = self.image_backbone(image)
            else:
                image_embedding = self.image_backbone(image)
            features.append(self.image_head(image_embedding))
        if self.metadata_head is not None:
            features.append(self.metadata_head(metadata))
        return self.classifier(torch.cat(features, dim=1))
