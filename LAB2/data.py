"""CIFAR-10 validation, the saved split and shared preprocessing."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from model import WEIGHTS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "prepared"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
AUGMENTATION = dict(crop_size=224, crop_scale=[0.70, 1.0], crop_ratio=[0.90, 1.10],
                    flip_p=0.5, jitter_p=0.8, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05,
                    erase_p=0.25, erase_scale=[0.02, 0.12], erase_ratio=[0.5, 2.0], erase_value=0,
                    order=["random_resized_crop", "horizontal_flip", "color_jitter", "to_tensor", "normalize", "erasing"])


def transform_spec(architecture):
    p = WEIGHTS[architecture].transforms()
    return dict(resize_size=p.resize_size, crop_size=p.crop_size, mean=p.mean, std=p.std,
                interpolation=p.interpolation.value, antialias=p.antialias, channels=3, dtype="float32")


def train_transform(architecture):
    p, a = WEIGHTS[architecture].transforms(), AUGMENTATION
    return T.Compose([
        T.RandomResizedCrop(a['crop_size'], scale=a['crop_scale'], ratio=a['crop_ratio'],
                            interpolation=p.interpolation, antialias=p.antialias),
        T.RandomHorizontalFlip(a['flip_p']),
        T.RandomApply([T.ColorJitter(a['brightness'], a['contrast'], a['saturation'], a['hue'])], p=a['jitter_p']),
        T.ToTensor(), T.Normalize(p.mean, p.std),
        T.RandomErasing(a['erase_p'], a['erase_scale'], a['erase_ratio'], a['erase_value'])])


def load_split():
    with np.load(BASE_DIR / "data/splits.npz", allow_pickle=False) as z:
        train, val = z['train_indices'], z['val_indices']
    for indices, count in [(train, 45000), (val, 5000)]:
        if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Invalid split shape/dtype")
    both = np.concatenate([train, val])
    if not np.array_equal(np.sort(both), np.arange(50000)):
        raise ValueError("Split indices must cover the training dataset once without overlap")
    return train, val, hashlib.sha256(both.astype('<i8').tobytes()).hexdigest()


def load_dataset(root=DATA_DIR, *, train, transform=None):
    ds = datasets.CIFAR10(str(root), train=train, transform=transform, download=False)
    count = 50000 if train else 10000
    if (ds.classes != CLASSES or ds.data.shape != (count, 32, 32, 3) or ds.data.dtype != np.uint8
            or not np.all(np.bincount(ds.targets, minlength=10) == count // 10)):
        raise ValueError("Invalid CIFAR-10 data, labels or class order")
    return ds


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def training_loaders(config, root, device, generator, smoke=False):
    train_idx, val_idx, identity = load_split()
    train = load_dataset(root, train=True, transform=train_transform(config['architecture']))
    val = load_dataset(root, train=True, transform=WEIGHTS[config['architecture']].transforms())
    stats = json.loads((BASE_DIR / 'reports/eda_stats.json').read_text(encoding='utf-8'))
    if stats['dataset'] != 'CIFAR-10' or stats['classes'] != CLASSES:
        raise ValueError("EDA metadata differs from dataset")
    for idx, key in [(train_idx, 'train_class_counts'), (val_idx, 'val_class_counts')]:
        if np.bincount(np.asarray(train.targets)[idx], minlength=10).tolist() != stats['metrics']['split'][key]:
            raise ValueError("Class counts differ from EDA")
    if smoke:
        train_idx, val_idx = train_idx[:config['batch_size'] * 2], val_idx[:32]
    kwargs = dict(num_workers=config['workers'], pin_memory=device.type == 'cuda',
                  drop_last=False, worker_init_fn=seed_worker)
    return (DataLoader(Subset(train, train_idx.tolist()), batch_size=config['batch_size'],
                       shuffle=True, generator=generator, **kwargs),
            DataLoader(Subset(val, val_idx.tolist()), batch_size=config['val_batch_size'],
                       shuffle=False, **kwargs), identity)


def check_metadata(state, *, smoke=False):
    architecture = state['architecture']
    if (state['dataset'] != 'CIFAR-10' or state['classes'] != CLASSES
            or state['weights_enum'] != str(WEIGHTS[architecture])
            or state['transform_spec'] != transform_spec(architecture)
            or state['augmentation_spec'] != AUGMENTATION
            or state['split_sha256'] != load_split()[2]
            or state['is_smoke'] != smoke
            or state['mode'] != ('fine_tune' if state['config']['fine_tune'] else 'feature_extract')):
        raise ValueError("Checkpoint data/preprocessing/mode/smoke mismatch")


def augmentation_preview(path, root, architecture="resnet18"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    ds = load_dataset(root, train=True)
    indices, _, _ = load_split()
    labels = np.asarray(ds.targets)
    transform, preset = train_transform(architecture), WEIGHTS[architecture].transforms()
    state = torch.get_rng_state()
    torch.manual_seed(42)
    fig, axes = plt.subplots(10, 4, figsize=(9, 20))
    try:
        for k, name in enumerate(CLASSES):
            image = ds.data[indices[labels[indices] == k][0]]
            axes[k, 0].imshow(image)
            axes[k, 0].set_title(name + ' / original')
            for ax in axes[k, 1:]:
                x = transform(Image.fromarray(image))
                x = x * torch.tensor(preset.std)[:, None, None] + torch.tensor(preset.mean)[:, None, None]
                ax.imshow(x.permute(1, 2, 0).clamp(0, 1).numpy())
            for ax in axes[k]:
                ax.axis('off')
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
        torch.set_rng_state(state)
