
import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn.functional as F

from torch.optim import AdamW
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

from train.arcface.network import ArcFaceEncoder



import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        self.margin = margin
        self.scale = scale

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight, dim=1),
        ).clamp(-1.0, 1.0)

        sine = torch.sqrt(
            (1.0 - cosine.square()).clamp(min=1e-7)
        )

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.mm,
        )

        one_hot = F.one_hot(
            labels, num_classes=self.weight.size(0)
        ).to(dtype=cosine.dtype)

        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return logits * self.scale



class ArcFaceModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        weight_decay: float = 1e-4,
        model_name: str = "mobilenet_v3",
        pretrained: bool = True,
        margin: float = 0.5,
        scale: float = 64.0,
        distance_threshold: float = 0.25,
        encoder_learning_rate: float = 1e-5,
        head_learning_rate: float = 1e-3,
        freeze_encoder_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ArcFaceEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            in_channels=1,
        )

        self._set_encoder_trainable(False)


        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

        self.distance_threshold = distance_threshold
        self._val_pair_outputs = []
        self._test_pair_outputs = []
    
    def _set_encoder_trainable(self, trainable: bool):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if trainable:
            self.encoder.train()
        else:
            self.encoder.eval()
    
    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def forward(self, images):
        """Возвращает эмбеддинги для сравнения изображений."""
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        # Формат обучающего batch: (images, class_labels)
        images, labels = batch

        embeddings = self.encoder(images)
        logits = self.arcface(embeddings, labels)

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

        return loss
    
    def on_train_epoch_start(self):
        trainable = self.current_epoch >= self.hparams.freeze_encoder_epochs
        self._set_encoder_trainable(trainable)

    def on_validation_epoch_start(self):
        self._val_pair_outputs.clear()

    def on_test_epoch_start(self):
        self._test_pair_outputs.clear()

    def _pair_step(self, batch, stage: str):
        # Формат pair batch: (image1, image2, same_label)
        image1, image2, labels = batch

        emb1 = self.encoder(image1)
        emb2 = self.encoder(image2)

        distances = torch.linalg.vector_norm(
            emb1 - emb2,
            ord=2,
            dim=1,
        )

        outputs = (
            distances.detach().cpu(),
            labels.detach().view(-1).cpu(),
        )

        if stage == "val":
            self._val_pair_outputs.append(outputs)
        else:
            self._test_pair_outputs.append(outputs)

    def validation_step(self, batch, batch_idx):
        self._pair_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._pair_step(batch, "test")

    def _compute_pair_metrics(self, outputs, stage: str):
        if not outputs:
            return

        distances = torch.cat([x[0] for x in outputs]).numpy()
        labels = torch.cat([x[1] for x in outputs]).numpy().astype(int)

        same_distances = distances[labels == 1]
        different_distances = distances[labels == 0]

        self.print(
            f"\n[{stage}] "
            f"same: mean={same_distances.mean():.4f}, "
            f"median={np.median(same_distances):.4f}; "
            f"different: mean={different_distances.mean():.4f}, "
            f"median={np.median(different_distances):.4f}"
        )

        self.print(
            f"[{stage}] "
            f"AUC(-distance)={roc_auc_score(labels, -distances):.4f}; "
            f"AUC(distance)={roc_auc_score(labels, distances):.4f}"
        )

        # ROC-AUC требует наличия обоих классов.
        if len(np.unique(labels)) < 2:
            self.print(
                f"{stage}: ROC-AUC cannot be calculated "
                "because only one class is present."
            )
            return

        # Чем меньше расстояние, тем вероятнее, что это одна печать.
        scores = -distances
        predictions = (
            distances < self.distance_threshold
        ).astype(int)

        # labels: 1 = same physical print, 0 = different prints.
        positives = labels == 1
        negatives = labels == 0

        far = (
            np.mean(predictions[negatives] == 1)
            if negatives.any()
            else float("nan")
        )
        frr = (
            np.mean(predictions[positives] == 0)
            if positives.any()
            else float("nan")
        )

        balanced_acc = (
            (1.0 - frr) + (1.0 - far)
        ) / 2.0

        f1 = f1_score(
            labels, predictions, zero_division=0
        )
        weighted_f1 = f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        precision = precision_score(
            labels, predictions, zero_division=0
        )
        recall = recall_score(
            labels, predictions, zero_division=0
        )
        roc_auc = roc_auc_score(labels, scores)

        # EER — приблизительная точка равенства FAR и FRR.
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

        metrics = {
            "roc_auc": roc_auc,
            "far": far,
            "frr": frr,
            "balanced_acc": balanced_acc,
            "eer": eer,
            "f1": f1,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
        }

        for name, value in metrics.items():
            self.log(
                f"{stage}_{name}",
                torch.tensor(
                    value,
                    dtype=torch.float32,
                    device=self.device,
                ),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self):
        self._compute_pair_metrics(
            self._val_pair_outputs, "val"
        )

    def on_test_epoch_end(self):
        self._compute_pair_metrics(
            self._test_pair_outputs, "test"
        )

    def configure_optimizers(self):
        return AdamW(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": self.hparams.encoder_learning_rate,
                    "name": "encoder",
                },
                {
                    "params": self.arcface.parameters(),
                    "lr": self.hparams.head_learning_rate,
                    "name": "arcface_head",
                },
            ],
            weight_decay=self.hparams.weight_decay,
        )
