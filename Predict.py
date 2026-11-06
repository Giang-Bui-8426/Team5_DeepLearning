from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,
    precision_score,recall_score,roc_auc_score,roc_curve,auc,)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ModelMLP import MLP


# Paths are resolved relative to this file, not the current terminal directory.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = ROOT / "checkpoints" / "checkpoint_best_global.pt"
REPORT_DIR = ROOT / "reports" / "test"

BATCH_SIZE = 256
NUM_WORKERS = 0
NUM_SAMPLES_TO_SHOW = 12
SEED = 42

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the models first or update CHECKPOINT_PATH."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]

    model = MLP(
        hidden=tuple(config["hidden"]),
        activation=config["activation"],
        normalization=config["normalization"],
        dropout=config["dropout"],
        residual_blocks=config["residual_blocks"],
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    labels_list, predictions_list, probabilities_list = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # [B, 1, 28, 28]
        labels = labels.to(device, non_blocking=True)  # [B]

        logits = model(images)  # [B, 10]
        total_loss += criterion(logits, labels).item()
        probabilities = torch.softmax(logits, dim=1)  # [B, 10]

        labels_list.append(labels.cpu())
        predictions_list.append(logits.argmax(dim=1).cpu())
        probabilities_list.append(probabilities.cpu())

    y_true = torch.cat(labels_list).numpy()
    y_pred = torch.cat(predictions_list).numpy()
    y_prob = torch.cat(probabilities_list).numpy()
    n_samples = len(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    false_positives = cm.sum(axis=0) - np.diag(cm)
    true_negatives = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    per_class_fpr = np.divide(
        false_positives,
        false_positives + true_negatives,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=(false_positives + true_negatives) != 0,
    )

    binary_labels = label_binarize(y_true, classes=range(NUM_CLASSES))
    scores = {
        "samples": n_samples,
        "loss": float(total_loss / n_samples),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr_macro": float(per_class_fpr.mean()),
        "auc_macro_ovr": float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=range(NUM_CLASSES)
        )),
        "auc_micro_ovr": float(roc_auc_score(binary_labels, y_prob, average="micro")),
    }
    return {"metrics": scores, "labels": y_true, "predictions": y_pred,
            "probabilities": y_prob, "confusion_matrix": cm}


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of images")
    ax.set(
        title="FashionMNIST — Test Confusion Matrix",
        xlabel="Predicted class", ylabel="Actual class",
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
    )
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    threshold = cm.max() / 2
    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            value = int(cm[row, col])
            ax.text(col, row, str(value), ha="center", va="center",
                    fontsize=8, color="white" if value > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    y_binary = label_binarize(y_true, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 7))
    for class_id, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_binary[:, class_id], y_prob[:, class_id])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{class_name}: {class_auc:.3f}")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False Positive Rate",
           ylabel="True Positive Rate", title="FashionMNIST — Test ROC (One-vs-Rest)")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(dataset, result, path):
    y_true = result["labels"]
    y_pred = result["predictions"]
    y_prob = result["probabilities"]
    rng = np.random.default_rng(SEED)

    correct = np.flatnonzero(y_true == y_pred)
    incorrect = np.flatnonzero(y_true != y_pred)
    n_wrong = min(len(incorrect), NUM_SAMPLES_TO_SHOW // 2)
    n_right = min(len(correct), NUM_SAMPLES_TO_SHOW - n_wrong)
    selected = np.concatenate([
        rng.choice(correct, n_right, replace=False),
        rng.choice(incorrect, n_wrong, replace=False),
    ])
    rng.shuffle(selected)

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, index in zip(axes.flat, selected):
        # FashionMNIST.data is the original uint8 image, before normalization.
        ax.imshow(dataset.data[int(index)].numpy(), cmap="gray", vmin=0, vmax=255)
        correct_prediction = bool(y_true[index] == y_pred[index])
        confidence = float(y_prob[index, y_pred[index]])
        ax.set_title(f"True: {CLASS_NAMES[y_true[index]]}", fontsize=10)
        ax.set_xlabel(f"Pred: {CLASS_NAMES[y_pred[index]]} ({confidence:.1%})",
                      color="forestgreen" if correct_prediction else "firebrick", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Test predictions — correct and incorrect examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(CHECKPOINT_PATH, device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Exactly as in Train.py
    ])
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    result = evaluate(model, test_loader, device)
    scores = result["metrics"]

    print(f"Model      : {checkpoint.get('model_name', 'unknown')}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Device     : {device}")
    print(f"Test images: {scores['samples']}")
    print("\nFinal test metrics (NOT validation metrics)")
    for name, value in scores.items():
        if name != "samples":
            print(f"  {name:<18}: {value:.4f}")
    print("\nPer-class classification report")
    print(classification_report(result["labels"], result["predictions"],
                                labels=range(NUM_CLASSES), target_names=CLASS_NAMES,
                                digits=4, zero_division=0))

    with (REPORT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({
            "model": checkpoint.get("model_name", "unknown"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "dataset": "FashionMNIST official test split",
            "metrics": scores,
            "classification_report": classification_report(
                result["labels"], result["predictions"], labels=range(NUM_CLASSES),
                target_names=CLASS_NAMES, output_dict=True, zero_division=0
            ),
        }, file, indent=2, ensure_ascii=False)

    with (REPORT_DIR / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_index", "actual", "predicted", "confidence", "correct"])
        for i, (actual, predicted, prob) in enumerate(zip(
            result["labels"], result["predictions"], result["probabilities"]
        )):
            writer.writerow([
                i, CLASS_NAMES[int(actual)], CLASS_NAMES[int(predicted)],
                f"{prob[int(predicted)]:.6f}", int(actual == predicted),
            ])

    plot_confusion_matrix(result["confusion_matrix"], REPORT_DIR / "confusion_matrix.png")
    plot_roc_curves(result["labels"], result["probabilities"], REPORT_DIR / "roc_ovr.png")
    plot_predictions(test_dataset, result, REPORT_DIR / "prediction_examples.png")
    print(f"\nReports saved to: {REPORT_DIR}")
