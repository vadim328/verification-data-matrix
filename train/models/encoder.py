from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingEncoder(nn.Module):
    """Image encoder shared by training wrappers and inference code."""

    def __init__(
        self,
        model_name: str = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )

        feature_dim = getattr(
            self.backbone,
            "head_hidden_size",
            self.backbone.num_features,
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    @classmethod
    def from_lightning_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_name: str | None = None,
        embedding_dim: int | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "EmbeddingEncoder":
        """Load only ``encoder.*`` weights from a Lightning checkpoint.

        ``model_name`` and ``embedding_dim`` override checkpoint metadata.
        They are required when the checkpoint does not contain matching
        Lightning hyperparameters.
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must contain a mapping")

        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint state_dict must be a mapping")

        hyperparameters: dict[str, Any] = checkpoint.get(
            "hyper_parameters", {}
        )
        resolved_model_name = model_name or hyperparameters.get("model_name")
        resolved_embedding_dim = (
            embedding_dim
            if embedding_dim is not None
            else hyperparameters.get("embedding_dim")
        )
        if resolved_model_name is None:
            raise ValueError(
                "model_name is required because checkpoint metadata "
                "does not define it"
            )
        if resolved_embedding_dim is None:
            raise ValueError(
                "embedding_dim is required because checkpoint metadata "
                "does not define it"
            )

        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(
                "Checkpoint does not contain weights with prefix 'encoder.'"
            )

        model = cls(
            model_name=resolved_model_name,
            embedding_dim=int(resolved_embedding_dim),
            pretrained=False,
            in_channels=1,
        )
        model.load_state_dict(encoder_state, strict=True)
        return model
