import csv
import os
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.tensorboard import SummaryWriter
from ModelMLP import MLP
from tqdm.autonotebook import tqdm
import shutil
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    recall_score, precision_score, f1_score, roc_auc_score,roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, RandomErasing

DATA_DIR = Path("./data")
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./logs")
REPORT_DIR = Path("./reports")

Seed = 42
EPOCHS = 30
NUM_WORKERS = 0
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
device = "cuda" if torch.cuda.is_available() else "cpu"

Config_parameter= [
    dict(name="01_baseline", hidden=(256, 128), activation="relu",
             normalization="none", dropout=0.1, residual_blocks=0,
             lr=1e-3, weight_decay=0.0, batch_size=64, label_smoothing=0.0,
             scheduler="none"),
    dict(name="02_wide_mlp", hidden=(1024, 512), activation="relu",
            normalization="none", dropout=0.1, residual_blocks=0,
            lr=1e-3, weight_decay=0.0, batch_size=64, label_smoothing=0.0,
            scheduler="cosine"),
    dict(name="03_deep_mlp", hidden=(512, 256, 128), activation="gelu",
            normalization="none", dropout=0.1, residual_blocks=0,
            lr=1e-3, weight_decay=0.0, batch_size=64, label_smoothing=0.0,
            scheduler="cosine"),
    dict(name="04_batchnorm_mlp", hidden=(512, 256, 128), activation="gelu",
            normalization="batchnorm", dropout=0.1, residual_blocks=0,
            lr=1e-3, weight_decay=0.0, batch_size=64, label_smoothing=0.0,
            scheduler="cosine"),
    dict(name="05_medium_mlp", hidden=(1024, 512, 256), activation="gelu",
            normalization="batchnorm", dropout=0.15, residual_blocks=0,
            lr=2e-3, weight_decay=1e-2, batch_size=64, label_smoothing=0.0,
            scheduler="cosine"),
    dict(name="06_residual_mlp", hidden=(256,), activation="gelu",
            normalization="layernorm", dropout=0.2, residual_blocks=2,
            lr=5e-4, weight_decay=1e-3, batch_size=64, label_smoothing=0.00,
            scheduler="cosine"),
]
def plot_multiclass_roc_curve(labels, all_probs, epoch, writer):
    # Binarize labels cho bài toán multi-class
    y_bin = label_binarize(labels, classes=range(10))
    all_probs_np = np.array(all_probs)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Tính TPR, FPR và vẽ cho từng class
    for i in range(10):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], all_probs_np[:, i])
        roc_auc_i = auc(fpr_i, tpr_i)
        ax.plot(fpr_i, tpr_i, lw=1.5, alpha=0.6, label=f'Class {i} (AUC = {roc_auc_i:.3f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title(f'ROC Curve - Epoch {epoch}', fontsize=14, fontweight='bold')
    
    # Đặt legend ở ngoài để không che đồ thị
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9) 
    fig.tight_layout()
    
    # Ghi vào TensorBoard
    writer.add_figure("ROC_Curve", fig, epoch)
    plt.close(fig)
# output average loss
def train(model,dataloader,optimizer,loss_function, epoch,writer):
    model.train()
    running_loss = 0.0
    images = 0
    progress_bar = tqdm(dataloader,colour="green")
    for iter, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss_value = loss_function(pred, y)
        
        optimizer.zero_grad()
        loss_value.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss_value.item() * y.size(0)
        images += y.size(0)
        if iter % 100 == 0:
            progress_bar.set_description("Epoch {}/{}. Loss {:.3f}".format(epoch+1, EPOCHS, loss_value))
            writer.add_scalar("Loss/train/batch", loss_value.item(), iter + epoch * len(dataloader))
    avg_loss = running_loss / images
    return avg_loss

@torch.no_grad()
def validate(model, dataloader, loss_function,writer=None, epoch=None):
    model.eval()

    total_loss = 0.0
    predictions = []
    labels = []
    all_probs = []
    total_samples = 0

    for X, y in dataloader:
        # X: [B, 1, 28, 28]
        # y: [B]
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # [B, 1, 28, 28] -> [B, 10]
        outputs = model(X)
        loss = loss_function(outputs, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = F.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().tolist())
        # [B, 10] -> [B]
        predicted = outputs.argmax(dim=1)

        predictions.extend(predicted.cpu().tolist())
        labels.extend(y.cpu().tolist())


    cm = confusion_matrix(labels, predictions, labels=range(10))

    fp = cm.sum(axis=0) - cm.diagonal()
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + cm.diagonal()
    fpr = (fp / (fp + tn)).mean()
    auc = roc_auc_score(label_binarize(labels, classes=range(10)), all_probs, average="macro", multi_class="ovr")

    if writer is not None and epoch is not None:
        plot_multiclass_roc_curve(labels, all_probs, epoch, writer)
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "fpr": float(fpr),
        "auc": float(auc),
    }

def save_checkpoint(model, optimizer, epoch,acc, path,config,metrics,scheduler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch+1,
        "config": config.copy(),
        "model_name": config["name"],
        "best_metrics": metrics.copy(),
        "best_acc": acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, path)

# output start epoch
def load_checkpoint(model, optimizer, path,scheduler):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    if scheduler_state_dict is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scheduler.load_state_dict(scheduler_state_dict)
    return checkpoint

def experiment_parameter(config,train_data,val_data):
    torch.manual_seed(Seed)
    print(f"Running experiment: {config['name']}")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,num_workers=NUM_WORKERS,
                                generator=torch.Generator().manual_seed(1))
    val_loader = DataLoader(val_data, num_workers=NUM_WORKERS,batch_size=256, shuffle=False)

    name_model = config["name"]
    model = MLP(hidden = config["hidden"],activation = config['activation'],
                normalization=config['normalization'],dropout=config['dropout'],
                residual_blocks=config['residual_blocks']).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                     if config["scheduler"] == "cosine" else None)
    model_dir = CHECKPOINT_DIR / name_model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    last_path = model_dir / "last_checkpoint.pt"
    best_path = model_dir / "checkpoint_best.pt"
    best_acc,best_metrics = 0,0
    start_epoch = 0
    if last_path.exists():
        checkpoint = load_checkpoint(model, optimizer, last_path,scheduler)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        best_metrics = checkpoint["best_metrics"]
    print(f"{name_model}: epoch {start_epoch}")
    best_loss, bad, patience = float("inf"), 0, 5
    with SummaryWriter(f"{LOG_DIR}/{config['name']}") as writer:
        writer.add_text("Config", str(config), 0)
        writer.add_scalar("Model/parameters", sum(p.numel() for p in model.parameters()), 0)
        for epoch in range(start_epoch, EPOCHS):
            torch.manual_seed(Seed + epoch)
            if device == "cuda":
                torch.cuda.manual_seed_all(Seed + epoch)
            train_loss = train(model, train_loader, optimizer, loss_fn,epoch,writer)
            val = validate(model, val_loader, nn.CrossEntropyLoss(),writer, epoch)

            if val["loss"] < best_loss - 1e-4:
                best_loss, bad = val["loss"], 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stop tại epoch {epoch + 1}")
                    break
            writer.add_scalar("Loss/train/epoch", train_loss, epoch)
            writer.add_scalar("Loss/validation", val["loss"], epoch)
            for key in ("accuracy", "precision", "recall", "f1", "fpr"):
                writer.add_scalar(f"Validation/{key}", val[key], epoch)
            writer.add_scalar("Optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            if scheduler:
                scheduler.step()
                
            if val["accuracy"] > best_acc:
                best_acc = val["accuracy"]
                best_metrics = val
                save_checkpoint(model, optimizer,epoch,best_acc,best_path,config,best_metrics,scheduler)
            save_checkpoint(model, optimizer, epoch,best_acc, last_path,config,best_metrics,scheduler)
    return {"name": name_model,**best_metrics}
def compare_models(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    keys = ("accuracy", "precision", "recall", "f1", "fpr", "auc")
    names = [result["name"] for result in results]

    with (REPORT_DIR / "best_models.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "loss", *keys])
        writer.writeheader()

        for result in results:
            writer.writerow({
                key: result[key]
                for key in ["name", "loss", *keys]
            })

    with SummaryWriter(log_dir=str(LOG_DIR / "comparison")) as writer:
        for key in keys:
            values = [result[key] for result in results]

            # Thu gọn figsize từ (12, 6) xuống (10, 4.5) để vừa vặn chiều dọc trên TensorBoard
            fig, ax = plt.subplots(figsize=(10, 4.5))

            bars = ax.barh(
                names,
                values,
                color=plt.cm.tab10.colors[:len(names)]
            )

            ax.invert_yaxis()
            ax.set_xlim(0, 1.08)
            ax.set_xlabel(key.upper(), fontsize=12)
            ax.set_title(
                f"Comparison of 5 MLP Models - {key.upper()}",
                fontsize=15,
                fontweight="bold"
            )

            ax.tick_params(axis="both", labelsize=10)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

            for bar, value in zip(bars, values):
                ax.text(
                    value + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}",
                    va="center",
                    fontsize=11,
                    fontweight="bold"
                )

            fig.tight_layout()

            fig.savefig(REPORT_DIR / f"comparison_{key}.png", dpi=200, bbox_inches="tight")
            writer.add_figure(f"Comparison/{key.upper()}", fig, global_step=0)

            for result in results:
                writer.add_scalar(f"Best/{key}/{result['name']}", result[key], 0)

            plt.close(fig)

        matrix = np.array([[result[key] for key in keys] for result in results])
        # Tương tự chỉnh lại form gọn hơn cho Heatmap
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        colors = np.zeros_like(matrix)
        for j, key in enumerate(keys):
            column = matrix[:, j]
            spread = column.max() - column.min()
            colors[:, j] = ((column.max() - column) if key == "fpr" else
                            (column - column.min())) / spread if spread > 0 else 0.5
                            
        image = ax.imshow(colors, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(keys)), [key.upper() for key in keys])
        ax.set_yticks(range(len(names)), names)
        ax.tick_params(labelsize=11)
        ax.set_title("Best Validation Metrics - All 5 MLP Models", fontsize=15, pad=12)
        
        for i in range(len(names)):
            for j in range(len(keys)):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        color="white" if colors[i, j] > 0.55 else "black", fontsize=10)
                        
        fig.colorbar(image, ax=ax, shrink=0.8, label="Relative performance (higher is better)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "comparison_heatmap.png", dpi=200, bbox_inches="tight")
        writer.add_figure("Comparison/All_metrics_heatmap", fig, 0)
        plt.close(fig)

        # Giảm chiều cao phần Accuracy vs Loss
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for result in results:
            ax.scatter(result["loss"], result["accuracy"], s=120)
            ax.annotate(result["name"], (result["loss"], result["accuracy"]),
                        xytext=(7, 7), textcoords="offset points", fontsize=9)
        ax.set_xlabel("Validation Cross-Entropy Loss (lower is better)", fontsize=11)
        ax.set_ylabel("Validation Accuracy (higher is better)", fontsize=11)
        ax.set_title("Accuracy vs. Validation Loss", fontsize=15)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "comparison_accuracy_vs_loss.png", dpi=200, bbox_inches="tight")
        writer.add_figure("Comparison/Accuracy_vs_loss", fig, 0)
        plt.close(fig)

    best = max(results, key=lambda x: x["accuracy"])

    print(f"\nBest Model: {best['name']}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"F1 Score: {best['f1']:.4f}")
    print(f"AUC Score: {best['auc']:.4f}")

    return best


if __name__ == "__main__":
    best_acc = 0.0
    start_epoch = 0
    print(f"{device}")
    train_transform = Compose([RandomHorizontalFlip(),ToTensor(), Normalize((0.5,), (0.5,))])
    eval_transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_data = datasets.FashionMNIST(root=DATA_DIR,train=True,download=True,transform=train_transform)
    eval_data  = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=eval_transform)
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(train_data), generator=g).tolist()
    train_data = Subset(train_data, perm[:50000])
    evaluate_data = Subset(eval_data, perm[50000:])

    print(f"Số lượng ảnh train: {len(train_data)}, validation: {len(evaluate_data)}")

    writer = SummaryWriter(LOG_DIR)
    
    results = [experiment_parameter(cfg,train_data,evaluate_data)
                   for cfg in Config_parameter]
    
    best = compare_models(results)
    source = CHECKPOINT_DIR / best["name"] / "checkpoint_best.pt"
    destination = CHECKPOINT_DIR / "checkpoint_best_global.pt"
    shutil.copy2(source, destination)
    writer.close()
    