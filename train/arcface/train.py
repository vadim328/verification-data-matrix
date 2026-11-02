
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from arcface.dataset import PrintDataModule
from arcface.model import ArcFaceModel
from utils import get_max_image_size


DATA_DIR = Path("data/dataset")
OUTPUT_DIR = Path("checkpoints")

BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 2e-3
EMBEDDING_DIM = 128


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(42, workers=True)

    max_width, max_height = get_max_image_size(DATA_DIR)

    data_module = PrintDataModule(
        data_dir=DATA_DIR,
        max_width=max_width,
        max_height=max_height,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Создаём датасеты, чтобы получить количество классов.
    data_module.setup("fit")
    num_classes = data_module.num_classes

    print(f"Number of ArcFace classes: {num_classes}")
    print(f"Number of train images: {len(data_module.train_dataset)}")
    print(f"Number of validation pairs: {len(data_module.val_dataset)}")

    model = ArcFaceModel(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        model_name="mobilenetv4_hybrid_large",  # либо "mobilenet_v3"
        pretrained=True,
        margin=0.5,
        scale=64.0,
        distance_threshold=0.25,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="arcface-{epoch:02d}-{val_roc_auc:.4f}",
        monitor="val_roc_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_roc_auc",
        mode="max",
        patience=5,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data_module)

    print("Best checkpoint:", checkpoint_callback.best_model_path)

    test_results = trainer.test(
        model,
        datamodule=data_module,
        ckpt_path="best",
    )

    print("Test results:", test_results)


if __name__ == "__main__":
    main()
