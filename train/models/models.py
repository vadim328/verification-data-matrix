import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from train.models.encoder import EmbeddingEncoder


# Backward-compatible import for code which used train.models.models.
EmbeddingTimm = EmbeddingEncoder


class EmbeddingResNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )
        backbone = models.resnet18(weights=weights)

        # Заменяем RGB-вход на grayscale.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            # Переносим предобученные веса RGB-фильтров
            # в одноканальный слой.
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        backbone.conv1 = new_conv

        # Убираем классификационную голову ImageNet.
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projection(x)

        # Нормализуем эмбеддинги для contrastive loss.
        return F.normalize(x, p=2, dim=1)


class EmbeddingMobileNetV3(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )
        backbone = models.mobilenet_v3_large(weights=weights)

        # Заменяем RGB-вход на grayscale.
        old_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )

        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

        backbone.features[0][0] = new_conv

        # Убираем классификационную голову ImageNet.
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projection(x)

        return F.normalize(x, p=2, dim=1)
