
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc

from train.contrastive.network import ContrastiveEncoder


class SiameseModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 128,
        learning_rate: float = 1e-3,
        margin: float = 1.0,
        weight_decay: float = 1e-4,
        model_name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
    ):
        super().__init__()
        if not 0.0 < margin <= 2.0:
            raise ValueError("margin must be in (0, 2] for cosine distance")

        self.save_hyperparameters()

        self.encoder = ContrastiveEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            in_channels=1,
        )
        
        # Для cosine distance положительные пары стремятся к 0, а
        # отрицательные — как минимум к margin. Половина margin является
        # разумным стартовым порогом до первой полной validation-эпохи.
        # Buffer сохраняется в checkpoint вместе с выбранным на val порогом.
        self.register_buffer(
            "decision_threshold",
            torch.tensor(margin / 2, dtype=torch.float32),
        )

        self._val_distances = []
        self._val_labels = []

        self._test_distances = []
        self._test_labels = []

        # Train balanced accuracy считаем глобально, не сохраняя все
        # предсказания и не перенося каждый batch на CPU.
        self.register_buffer(
            "_train_confusion",
            torch.zeros(4, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def contrastive_loss(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        similarities = F.cosine_similarity(
            embedding1,
            embedding2,
            dim=1,
        )
        distances = 1.0 - similarities

        # label=1 (same): distance -> 0.
        # label=0 (different): distance >= margin.
        losses = torch.where(
            labels == 1,
            distances.square(),
            F.relu(self.hparams.margin - distances).square(),
        )

        return losses.mean()

    def _shared_step(self, batch, stage: str):
        image1, image2, labels = batch
        labels = labels.view(-1).long()

        embedding1 = self(image1)
        embedding2 = self(image2)

        loss = self.contrastive_loss(
            embedding1,
            embedding2,
            labels,
        )

        similarities = F.cosine_similarity(
            embedding1,
            embedding2,
            dim=1,
        )
        distances = 1.0 - similarities

        if stage in ("val", "test"):
            getattr(self, f"_{stage}_distances").append(
                distances.detach().cpu()
            )
            getattr(self, f"_{stage}_labels").append(
                labels.detach().cpu()
            )

        if stage == "train":
            predictions = distances < self.decision_threshold
            positive_mask = labels == 1
            negative_mask = labels == 0
            self._train_confusion += torch.stack([
                (predictions & positive_mask).sum(),
                positive_mask.sum(),
                ((~predictions) & negative_mask).sum(),
                negative_mask.sum(),
            ])

        batch_size = labels.size(0)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    @staticmethod
    def _balanced_accuracy(
        true_positive: torch.Tensor,
        positive_count: torch.Tensor,
        true_negative: torch.Tensor,
        negative_count: torch.Tensor,
    ) -> torch.Tensor | None:
        recalls = []
        if positive_count.item() > 0:
            recalls.append(true_positive.float() / positive_count)
        if negative_count.item() > 0:
            recalls.append(true_negative.float() / negative_count)
        if not recalls:
            return None
        return torch.stack(recalls).mean()

    @staticmethod
    def _best_threshold(
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor | None:
        positive_count = (labels == 1).sum()
        negative_count = (labels == 0).sum()
        if positive_count.item() == 0 or negative_count.item() == 0:
            return None

        sorted_distances, order = distances.sort()
        sorted_labels = labels[order]

        # Порог distance < threshold не может разделить одинаковые distance,
        # поэтому рассматриваем только границы между уникальными значениями.
        _, counts = torch.unique_consecutive(
            sorted_distances,
            return_counts=True,
        )
        group_ends = counts.cumsum(0) - 1

        cumulative_positive = (sorted_labels == 1).cumsum(0)
        cumulative_negative = (sorted_labels == 0).cumsum(0)

        true_positive = torch.cat([
            cumulative_positive.new_zeros(1),
            cumulative_positive[group_ends],
        ])
        false_positive = torch.cat([
            cumulative_negative.new_zeros(1),
            cumulative_negative[group_ends],
        ])

        balanced_accuracy = 0.5 * (
            true_positive.float() / positive_count
            + (negative_count - false_positive).float() / negative_count
        )
        best_boundary = int(balanced_accuracy.argmax())

        unique_distances = sorted_distances[group_ends]
        if best_boundary == 0:
            # При строгом "<" threshold=min(distance) не принимает ни одну
            # пару, включая повторы минимального distance.
            return unique_distances[0]
        if best_boundary == unique_distances.numel():
            return torch.nextafter(
                unique_distances[-1],
                torch.tensor(
                    float("inf"),
                    dtype=unique_distances.dtype,
                    device=unique_distances.device,
                ),
            )

        left = unique_distances[best_boundary - 1]
        right = unique_distances[best_boundary]
        return left + (right - left) / 2

    def _log_epoch_metrics(
        self,
        stage: str,
        distances_list: list,
        labels_list: list,
        tune_threshold: bool,
    ):
        if not distances_list:
            return

        distances = torch.cat(distances_list)
        labels = torch.cat(labels_list).long()

        if tune_threshold:
            threshold = self._best_threshold(distances, labels)
            if threshold is not None:
                self.decision_threshold.copy_(
                    threshold.to(self.decision_threshold.device)
                )

        threshold = self.decision_threshold.detach().cpu()
        predictions = distances < threshold

        positive_mask = labels == 1  # Один экземпляр
        negative_mask = labels == 0  # Разные экземпляры
        true_positive = (predictions & positive_mask).sum()
        true_negative = ((~predictions) & negative_mask).sum()
        positive_count = positive_mask.sum()
        negative_count = negative_mask.sum()

        if positive_count.item() > 0 and negative_count.item() > 0:
            # Для label=1 (same) большая similarity является большим score.
            roc_auc = binary_auroc(1.0 - distances, labels)
            self.log(
                f"{stage}_roc_auc",
                roc_auc,
                prog_bar=True,
            )

        balanced_accuracy = self._balanced_accuracy(
            true_positive,
            positive_count,
            true_negative,
            negative_count,
        )
        if balanced_accuracy is not None:
            self.log(
                f"{stage}_balanced_acc",
                balanced_accuracy,
                prog_bar=True,
            )

        # FAR: разные экземпляры ошибочно признаны одинаковыми.
        if negative_mask.any():
            far = (negative_count - true_negative).float() / negative_count
            self.log(
                f"{stage}_far",
                far,
                prog_bar=True,
            )

        # FRR: один экземпляр ошибочно признан разными.
        if positive_mask.any():
            frr = (positive_count - true_positive).float() / positive_count
            self.log(
                f"{stage}_frr",
                frr,
                prog_bar=True,
            )

        self.log(
            f"{stage}_threshold",
            self.decision_threshold,
            prog_bar=False,
        )

    def on_train_epoch_start(self):
        self._train_confusion.zero_()

    def on_train_epoch_end(self):
        true_positive, positive_count, true_negative, negative_count = (
            self._train_confusion
        )
        balanced_accuracy = self._balanced_accuracy(
            true_positive,
            positive_count,
            true_negative,
            negative_count,
        )
        if balanced_accuracy is not None:
            self.log(
                "train_balanced_acc",
                balanced_accuracy,
                prog_bar=True,
            )
        self._train_confusion.zero_()

    def on_validation_epoch_start(self):
        self._val_distances.clear()
        self._val_labels.clear()


    def on_validation_epoch_end(self):
        try:
            self._log_epoch_metrics(
                "val",
                self._val_distances,
                self._val_labels,
                tune_threshold=True,
            )
        finally:
            self._val_distances.clear()
            self._val_labels.clear()

    def on_test_epoch_start(self):
        self._test_distances.clear()
        self._test_labels.clear()


    def on_test_epoch_end(self):
        try:
            self._log_epoch_metrics(
                "test",
                self._test_distances,
                self._test_labels,
                tune_threshold=False,
            )
        finally:
            self._test_distances.clear()
            self._test_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
