"""Practice 2 - Transfer learning with ResNet18, DenseNet121, VGG16.

Dataset: Fashion-MNIST (10 image classes). Results: TensorBoard logs + CSV +
model structure files + optional checkpoints.

Examples:
    python Train.py --quick
    python Train.py --models resnet18,densenet121,vgg16 --epochs 3
    python Train.py --models resnet18 --epochs 3 --fine-tune

Install: pip install torch torchvision tensorboard
"""

import argparse
import copy
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_MODELS = ("resnet18", "densenet121", "vgg16")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(root, batch_size, image_size, n_train, n_val, n_test, seed):
    # Resize CIFAR-10 32x32 images to the input scale of ImageNet models.
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_full = datasets.CIFAR10(root=root, train=True, download=True,
                                  transform=train_transform)
    val_full = datasets.CIFAR10(root=root, train=True, download=False,
                                transform=eval_transform)
    test_full = datasets.CIFAR10(root=root, train=False, download=True,
                                 transform=eval_transform)

    if n_train + n_val > len(train_full) or n_test > len(test_full):
        raise ValueError("Too many samples: CIFAR-10 has 50,000 train / 10,000 test images.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    test_perm = torch.randperm(len(test_full), generator=g).tolist()
    train_ds = Subset(train_full, perm[:n_train])
    val_ds = Subset(val_full, perm[n_train:n_train + n_val])
    test_ds = Subset(test_full, test_perm[:n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader




def parameter_statistics(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def write_architecture(model, name, run_dir):
    total, trainable = parameter_statistics(model)
    out_path = run_dir / f"architecture_{name}.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Model: {name}\nTotal parameters: {total:,}\n")
        f.write(f"Trainable parameters: {trainable:,}\n\n")
        f.write("Model structure:\n")
        f.write(str(model) + "\n\n")
        f.write("Parameter name | Shape | Number | Trainable\n")
        for param_name, p in model.named_parameters():
            f.write(f"{param_name} | {tuple(p.shape)} | {p.numel():,} | {p.requires_grad}\n")
    print(f"Model: {name} | Total: {total:,} | Trainable: {trainable:,}")
    print(f"Architecture saved to: {out_path}")


def run_epoch(model, loader, criterion, device, optimizer=None, feature_extract=False,
              head=None):
    is_training = optimizer is not None
    if is_training and feature_extract:
        # Keep the pretrained frozen backbone (including BatchNorm stats) fixed.
        model.eval()
        head.train()
    elif is_training:
        model.train()
        # Freeze running statistics of any frozen BatchNorm modules.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(
                p.requires_grad for p in module.parameters(recurse=False)
            ):
                module.eval()
    else:
        model.eval()

    loss_sum, correct, sample_count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)