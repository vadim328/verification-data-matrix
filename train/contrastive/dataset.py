
import random
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Union, Optional

import lightning.pytorch as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


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


class PadToSize:
    def __init__(self, size: tuple[int, int], fill: int = 255):
        self.height, self.width = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        if width > self.width or height > self.height:
            raise ValueError(
                f"Image size {width}x{height} exceeds "
                f"target size {self.width}x{self.height}"
            )

        left = (self.width - width) // 2
        top = (self.height - height) // 2
        right = self.width - width - left
        bottom = self.height - height - top

        return TF.pad(
            image,
            [left, top, right, bottom],
            fill=self.fill,
        )


class PrintPairDataset(Dataset):
    """
    Dataset для обучения эмбеддингов физических экземпляров QR-кодов.

    Возвращает:
        image1: Tensor [1, H, W]
        image2: Tensor [1, H, W]
        label:  Tensor [] — 1 для same, 0 для different
    """

    EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        max_width: int,
        max_height: int,
        augment: bool = True,
        include_self_pairs: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.augment = augment

        self.to_tensor = transforms.ToTensor()
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

        self.pairs = self._build_pairs(include_self_pairs)

        if not self.pairs:
            raise ValueError(
                f"No image pairs found in {self.root_dir}"
            )

    def _build_pairs(self, include_self_pairs: bool) -> list:
        pairs = []

        # Каждая папка верхнего уровня — отдельный QR-код.
        qr_dirs = sorted(
            path for path in self.root_dir.iterdir()
            if path.is_dir()
        )

        for qr_dir in qr_dirs:
            images = []

            for capture_type in ("normal", "flipped"):
                capture_dir = qr_dir / capture_type

                if not capture_dir.is_dir():
                    continue

                for image_path in sorted(capture_dir.iterdir()):
                    if image_path.suffix.lower() not in self.EXTENSIONS:
                        continue

                    # Буква A–J идентифицирует физический экземпляр.
                    images.append((image_path, image_path.stem))

            if include_self_pairs:
                combinations_iter = combinations_with_replacement(
                    images, 2
                )
            else:
                combinations_iter = combinations(images, 2)

            for (path1, print_id1), (path2, print_id2) in combinations_iter:
                label = int(print_id1 == print_id2)
                pairs.append((path1, path2, label))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("L").copy()

    def _should_augment_pair(
        self,
        path1: Path,
        path2: Path,
    ) -> tuple[bool, bool]:
        same_file = path1 == path2

        if same_file:
            # Исключаем тривиальную пару: оригинал + тот же оригинал.
            return random.choice([
                (True, False),
                (False, True),
                (True, True),
            ])

        # Независимый выбор для обоих изображений.
        return (
            random.random() < 0.5,
            random.random() < 0.5,
        )

    def _prepare_image(self, path, should_augment):
        image = self._load_image(path)
        if should_augment:
            # augmentation уже вызывает ToTensor()
            image = self.augmentation(image)
        else:
            image = self.to_tensor(image)
        return self.base_transform(image)


    def __getitem__(self, index: int):
        path1, path2, label = self.pairs[index]

        if self.augment:
            augment1, augment2 = self._should_augment_pair(
                path1, path2
            )
        else:
            augment1, augment2 = False, False

        image1 = self._prepare_image(path1, augment1)
        image2 = self._prepare_image(path2, augment2)

        return (
            image1,
            image2,
            torch.tensor(label, dtype=torch.float32),
        )


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
            self.train_dataset = PrintPairDataset(
                root_dir=self.data_dir / "train",
                max_width=self.max_width,
                max_height=self.max_height,
                augment=True,
                include_self_pairs=True,
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

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
