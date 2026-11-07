"""Fashion-MNIST data pipeline for Practice 2."""

import os
import gzip
import shutil
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import torch
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEED = 1
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
FASHION_MNIST_GITHUB_URL = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
)
FASHION_MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


TRAIN_TRANSFORM = build_train_transform()
EVAL_TRANSFORM = build_eval_transform()


def download_fashion_mnist(root: Union[str, PathLike[str]]) -> None:
    raw_dir = Path(root) / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [
        filename
        for filename in FASHION_MNIST_FILES
        if not (raw_dir / filename).is_file()
        or (raw_dir / filename).stat().st_size == 0
    ]
    if not missing_files:
        return

    for filename in missing_files:
        destination = raw_dir / filename
        temporary = destination.with_name(destination.name + ".tmp")
        urlretrieve(FASHION_MNIST_GITHUB_URL + filename, temporary)
        if temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded empty file for Fashion-MNIST: {filename}"
            )
        os.replace(temporary, destination)

    for filename in FASHION_MNIST_FILES:
        archive = raw_dir / filename
        extracted = raw_dir / filename.removesuffix(".gz")
        if archive.is_file() and not extracted.is_file():
            with gzip.open(archive, "rb") as source:
                with open(extracted, "wb") as target:
                    shutil.copyfileobj(source, target)


def open_fashion_mnist(
    root: Union[str, PathLike[str]],
    train: bool,
    transform: transforms.Compose,
) -> datasets.FashionMNIST:
    try:
        return datasets.FashionMNIST(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )
    except RuntimeError:
        try:
            download_fashion_mnist(root)
            return datasets.FashionMNIST(
                root=str(root),
                train=train,
                download=False,
                transform=transform,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                "Fashion-MNIST could not be downloaded from the torchvision "
                "mirror or the GitHub fallback."
            ) from fallback_error


def load_split_indices(
    split_path: Union[str, PathLike[str]],
    seed: int = SPLIT_SEED,
    n_train: int = 50_000,
    n_val: int = 10_000,
) -> tuple[list[int], list[int]]:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        try:
            split = torch.load(split_path, map_location="cpu", weights_only=True)
        except TypeError:
            split = torch.load(split_path, map_location="cpu")

        required = {"seed", "train_indices", "val_indices"}
        missing = required.difference(split)
        if missing:
            raise ValueError(
                f"Split file is missing fields: {sorted(missing)} in {split_path}"
            )
        if split["seed"] != seed:
            raise ValueError(
                f"Split seed mismatch: expected {seed}, found {split['seed']}"
            )

        train_indices = torch.as_tensor(
            split["train_indices"], dtype=torch.long
        ).tolist()
        val_indices = torch.as_tensor(
            split["val_indices"], dtype=torch.long
        ).tolist()
    else:
        indices = randperm(
            60_000,
            generator=Generator().manual_seed(seed),
        ).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        split = {
            "seed": seed,
            "train_indices": torch.tensor(train_indices, dtype=torch.long),
            "val_indices": torch.tensor(val_indices, dtype=torch.long),
        }

        temporary_path = split_path.with_name(split_path.name + ".tmp")
        torch.save(split, temporary_path)
        os.replace(temporary_path, split_path)

    if len(train_indices) != n_train or len(val_indices) != n_val:
        raise ValueError(
            "Saved split lengths do not match the requested "
            f"{n_train}/{n_val} train/validation sizes."
        )
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("Split contains duplicate training indices.")
    if len(set(val_indices)) != len(val_indices):
        raise ValueError("Split contains duplicate validation indices.")
    if len(set(train_indices).intersection(val_indices)) != 0:
        raise ValueError("Train and validation indices overlap.")
    if any(index < 0 or index >= 60_000 for index in train_indices + val_indices):
        raise ValueError("Split indices are outside the Fashion-MNIST train set.")

    expected_indices = randperm(
        60_000,
        generator=Generator().manual_seed(seed),
    ).tolist()
    expected_train = expected_indices[:n_train]
    expected_val = expected_indices[n_train:n_train + n_val]
    if train_indices != expected_train or val_indices != expected_val:
        raise ValueError(
            f"Split file does not match seed {seed} and the fixed 50k/10k split."
        )

    return train_indices, val_indices


def create_dataloaders(
    root: Union[str, PathLike[str]] = "data",
    batch_size: int = 32,
    image_size: int = 224,
    n_train: int = 50_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    seed: int = SPLIT_SEED,
    val_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    return_split_info: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader] | tuple[
    DataLoader, DataLoader, DataLoader, dict[str, object]
]:
    """Create separate train, validation, and official-test DataLoaders."""
    if image_size != 224:
        raise ValueError(
            "Practice 2 requires image_size=224 for the pretrained backbone."
        )
    if n_train != 50_000 or n_val != 10_000:
        raise ValueError(
            "Practice 2 requires a fixed 50,000/10,000 train/validation split."
        )
    if n_test != 10_000:
        raise ValueError("Practice 2 requires the full 10,000-image official test set.")

    root = Path(root)
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)
    train_source = open_fashion_mnist(root, train=True, transform=train_transform)
    val_source = open_fashion_mnist(root, train=True, transform=eval_transform)
    test_data = open_fashion_mnist(root, train=False, transform=eval_transform)

    if len(train_source) != 60_000:
        raise RuntimeError(
            f"Expected 60,000 Fashion-MNIST training images, found {len(train_source)}."
        )
    if len(test_data) != 10_000:
        raise RuntimeError(
            f"Expected 10,000 Fashion-MNIST test images, found {len(test_data)}."
        )

    split_path = root / "split_indices_seed1.pt"
    train_indices, val_indices = load_split_indices(
        split_path,
        seed=seed,
        n_train=n_train,
        n_val=n_val,
    )
    train_data = Subset(train_source, train_indices)
    val_data = Subset(val_source, val_indices)

    if val_batch_size is None:
        val_batch_size = batch_size
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=val_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    split_info = {
        "seed": seed,
        "split_path": str(split_path),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "image_size": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    if return_split_info:
        return train_loader, val_loader, test_loader, split_info
    return train_loader, val_loader, test_loader
