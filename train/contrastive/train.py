
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from train.contrastive.dataset import PrintDataModule
from train.contrastive.model import SiameseModel

DATA_DIR = Path("data/dataset")
OUTPUT_DIR = Path("checkpoints")

BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCHS = 2
LEARNING_RATE = 2e-5
EMBEDDING_DIM = 128


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(42, workers=True)

    max_width, max_height = 0, 0

    data_module = PrintDataModule(
        data_dir=DATA_DIR,
        max_width=max_width,
        max_height=max_height,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = SiameseModel(
        embedding_dim=EMBEDDING_DIM,
        learning_rate=LEARNING_RATE,
        margin=1.0,
        weight_decay=1e-4,
        model_name="mobilenetv4_hybrid_large",
        pretrained=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cuda",
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

    print(
        "Best checkpoint:",
        checkpoint_callback.best_model_path,
    )

    test_results = trainer.test(
        model,
        datamodule=data_module,
        ckpt_path="best",
    )

    print("Test results:", test_results)


if __name__ == "__main__":
    main()
