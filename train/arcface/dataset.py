
import random
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from train.contrastive.dataset import PrintPairDataset


class AddUnevenIllumination:
    def __init__(self, strength=0.15, p=0.3):
        self.strength = strength
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image: [C, H, W], диапазон [0, 1]
        if torch.rand(()) >= self.p:
            return image

        _, height, width = image.shape
        device, dtype = image.device, image.dtype

        y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        a = torch.empty((), device=device).uniform_(-1, 1)
        b = torch.empty((), device=device).uniform_(-1, 1)

        field = a * xx + b * yy
        field = field / field.abs().max().clamp_min(1e-6)

        factor = 1 + self.strength * field
        return (image * factor.unsqueeze(0)).clamp(0, 1)


class PrintImageDataset(Dataset):
    """Dataset отдельных изображений для обучения ArcFace."""

    EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        max_width: int,
        max_height: int,
        augment: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            ),
        ])

        self.augmentation = transforms.Compose([
            # Геометрия без поворотов
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.01, 0.01),
                    scale=(0.98, 1.02),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=255,
                ),
            ], p=0.3),

            # Яркость и контраст
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                ),
            ], p=0.4),

            # Размытие
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 0.5),
                ),
            ], p=0.15),

            transforms.ToTensor(),

            # Шум — с вероятностью 30%
            transforms.RandomApply([
                v2.GaussianNoise(sigma=0.02),
            ], p=0.3),

            # Неравномерное освещение — с вероятностью 30%
            transforms.RandomApply([
                AddUnevenIllumination(strength=0.15, p=1.0),
            ], p=0.3),

        ])

        self.samples = []
        self.class_to_idx = {}

        self._build_samples()

        if not self.samples:
            raise ValueError(
                f"No images found in {self.root_dir}"
            )

    def _build_samples(self):
        qr_dirs = sorted(
            path for path in self.root_dir.iterdir()
            if path.is_dir()
        )

        for qr_dir in qr_dirs:
            for capture_type in ("normal", "flipped"):
                capture_dir = qr_dir / capture_type

                if not capture_dir.is_dir():
                    continue

                image_paths = sorted(
                    path for path in capture_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in self.EXTENSIONS
                )

                for image_path in image_paths:
                    # Один класс на физический экземпляр.
                    # ID Data Matrix + буква A–J.
                    class_key = (
                        qr_dir.name,
                        image_path.stem,
                    )

                    if class_key not in self.class_to_idx:
                        self.class_to_idx[class_key] = (
                            len(self.class_to_idx)
                        )

                    label = self.class_to_idx[class_key]
                    self.samples.append((image_path, label))

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("L").copy()

        if self.augment:
            image = self.augmentation(image)

        image = self.base_transform(image)

        return image, torch.tensor(label, dtype=torch.long)



import lightning.pytorch as pl
import torch
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader


class PrintDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        max_width: int,
        max_height: int,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_width = max_width
        self.max_height = max_height

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = PrintImageDataset(
                root_dir=self.data_dir / "train",
                max_width=self.max_width,
                max_height=self.max_height,
                augment=True,
            )

            self.val_dataset = PrintPairDataset(
                root_dir=self.data_dir / "val",
                max_width=self.max_width,
                max_height=self.max_height,
                augment=False,
                include_self_pairs=False,
            )

        if stage in (None, "test"):
            self.test_dataset = PrintPairDataset(
                root_dir=self.data_dir / "test",
                max_width=self.max_width,
                max_height=self.max_height,
                augment=False,
                include_self_pairs=False,
            )

    @property
    def num_classes(self) -> int:
        return self.train_dataset.num_classes

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
