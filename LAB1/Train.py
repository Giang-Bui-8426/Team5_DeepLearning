import csv
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
from tqdm.autonotebook import tqdm

from ModelMLP import MLP
from TrainReport import RunRecorder, build_report, fig_confusion, fig_roc, load_runs

DATA_DIR = Path("./data")
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./logs")
REPORT_DIR = Path("./reports")

Seed = 42
EPOCHS = 30
NUM_WORKERS = 0
PATIENCE = 5            # early stopping: số epoch liên tiếp val loss không cải thiện
MAX_GRAD_NORM = 5.0     # ngưỡng clip gradient
LOG_EVERY = 50          # ghi loss theo batch (TensorBoard + báo cáo HTML) mỗi N iteration
FIG_EVERY = 5           # ghi hình confusion matrix / ROC / histogram trọng số mỗi N epoch
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
NUM_CLASSES = len(classes)
device = "cuda" if torch.cuda.is_available() else "cpu"

Config_parameter = [
    dict(name="01_baseline", hidden=(256, 128), activation="LeakyRelu",
         normalization="none", dropout=0.2, residual_blocks=0,
         lr=1e-3, weight_decay=1e-2, batch_size=64,
         scheduler="none"),
    dict(name="02_wide_mlp", hidden=(1024, 512), activation="LeakyRelu",
         normalization="none", dropout=0.2, residual_blocks=0,
         lr=1e-3, weight_decay=1e-2, batch_size=64,
         scheduler="cosine"),
    dict(name="03_deep_mlp", hidden=(512, 256, 128), activation="LeakyRelu",
         normalization="none", dropout=0.2, residual_blocks=0,
         lr=1e-3, weight_decay=1e-2, batch_size=64,
         scheduler="cosine"),
    dict(name="04_batchnorm_mlp", hidden=(512, 256, 128), activation="LeakyRelu",
         normalization="batchnorm", dropout=0.2, residual_blocks=0,
         lr=1e-3, weight_decay=1e-2, batch_size=64,
         scheduler="cosine"),
    dict(name="05_medium_mlp", hidden=(1024, 512, 256), activation="LeakyRelu",
         normalization="batchnorm", dropout=0.3, residual_blocks=0,
         lr=2e-3, weight_decay=1e-2, batch_size=64,
         scheduler="cosine"),
    dict(name="06_residual_mlp", hidden=(256,), activation="LeakyRelu",
         normalization="layernorm", dropout=0.3, residual_blocks=2,
         lr=5e-4, weight_decay=1e-2, batch_size=64,
         scheduler="cosine"),
]


def tensorboard_layout():
    """Tab 'Custom Scalars' của TensorBoard: gộp train/validation vào cùng một biểu đồ."""
    return {
        "Train vs Validation": {
            "Loss": ["Multiline", ["Loss/train/epoch", "Loss/validation"]],
            "Accuracy": ["Multiline", ["Accuracy/train", "Validation/accuracy"]],
            "Precision / Recall / F1": ["Multiline", ["Validation/precision", "Validation/recall", "Validation/f1"]],
        },
        "Optimizer": {
            "Learning rate": ["Multiline", ["Optimization/learning_rate"]],
            "Gradient norm": ["Multiline", ["Optimization/grad_norm_mean", "Optimization/grad_norm_max"]],
        },
        "Time": {
            "Seconds per epoch": ["Multiline", ["Time/train_seconds", "Time/validate_seconds"]],
        },
    }


def weight_norm(model):
    return float(torch.sqrt(sum(p.detach().float().pow(2).sum() for p in model.parameters())))


# output: dict thống kê của 1 epoch train
def train(model, dataloader, optimizer, loss_function, epoch, writer, recorder):
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    grad_norms = []
    images = 0
    progress_bar = tqdm(dataloader, colour="green")
    for it, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss_value = loss_function(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        grad_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  # norm TRƯỚC khi clip
        optimizer.step()

        batch = y.size(0)
        loss_sum += loss_value.detach() * batch
        correct += (pred.argmax(dim=1) == y).sum()
        grad_norms.append(torch.as_tensor(grad_norm, device=device).detach())
        images += batch

        if it % LOG_EVERY == 0:
            loss_item = loss_value.item()
            step = it + epoch * len(dataloader)
            progress_bar.set_description("Epoch {}/{}. Loss {:.3f}".format(epoch + 1, EPOCHS, loss_item))
            writer.add_scalar("Loss/train/batch", loss_item, step)
            writer.add_scalar("Optimization/grad_norm/batch", float(grad_norms[-1]), step)
            recorder.log_batch(epoch + 1, step, loss_item)

    g = torch.stack(grad_norms)
    return {
        "loss": loss_sum.item() / images,
        "accuracy": correct.item() / images,
        "grad_norm_mean": g.mean().item(),
        "grad_norm_max": g.max().item(),
        "clip_frac": (g > MAX_GRAD_NORM).float().mean().item(),
        "images": images,
    }


@torch.no_grad()
def validate(model, dataloader, loss_function):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_probs, all_labels = [], []

    for X, y in dataloader:
        # X: [B, 1, 28, 28]   y: [B]
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        outputs = model(X)                     # [B, 10]
        loss = loss_function(outputs, y)

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        all_probs.append(F.softmax(outputs, dim=1).cpu())
        all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy().astype(np.float64)
    y_pred = y_prob.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    fp = cm.sum(axis=0) - cm.diagonal()
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + cm.diagonal()
    fpr = np.mean(fp / np.maximum(fp + tn, 1))
    auc_macro = roc_auc_score(label_binarize(y_true, classes=np.arange(NUM_CLASSES)),
                              y_prob, average="macro")

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": float(fpr),
        "auc": float(auc_macro),
    }
    return metrics, y_true, y_prob


def save_checkpoint(model, optimizer, epoch, acc, path, config, metrics, scheduler=None, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch + 1,
        "config": dict(config),
        "model_name": config["name"],
        "best_metrics": dict(metrics),
        "best_acc": float(acc),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


# output: checkpoint (epoch đã hoàn tất nằm ở checkpoint["epoch"])
def load_checkpoint(model, optimizer, path, scheduler=None):
    # weights_only=False: đây là checkpoint do chính mình tạo (có dict config, metrics)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def log_tensorboard(writer, model, step, tr, val, e, lr, wnorm, t_train, t_val, figures, recorder):
    """Ghi toàn bộ chỉ số của 1 epoch lên TensorBoard. step = số epoch đã hoàn tất (1, 2, ...)."""
    writer.add_scalar("Loss/train/epoch", tr["loss"], step)
    writer.add_scalar("Loss/validation", val["loss"], step)
    writer.add_scalar("Loss/generalization_gap", val["loss"] - tr["loss"], step)
    writer.add_scalar("Accuracy/train", tr["accuracy"], step)
    for key in ("accuracy", "precision", "recall", "f1", "fpr", "auc"):
        writer.add_scalar(f"Validation/{key}", val[key], step)

    # trình tối ưu hoá
    writer.add_scalar("Optimization/learning_rate", lr, step)
    writer.add_scalar("Optimization/grad_norm_mean", tr["grad_norm_mean"], step)
    writer.add_scalar("Optimization/grad_norm_max", tr["grad_norm_max"], step)
    writer.add_scalar("Optimization/clip_fraction", tr["clip_frac"], step)
    writer.add_scalar("Optimization/weight_norm", wnorm, step)

    # thời gian
    writer.add_scalar("Time/train_seconds", t_train, step)
    writer.add_scalar("Time/validate_seconds", t_val, step)
    writer.add_scalar("Time/images_per_second", tr["images"] / max(t_train, 1e-9), step)

    # theo từng lớp
    for i, name in enumerate(classes):
        safe = name.replace("/", "_")
        writer.add_scalar(f"Class_F1/{safe}", e["pc"]["f1"][i], step)
        writer.add_scalar(f"Class_Recall/{safe}", e["pc"]["recall"][i], step)
        writer.add_scalar(f"Class_Precision/{safe}", e["pc"]["precision"][i], step)
        writer.add_scalar(f"Class_AUC/{safe}", e["pc"]["auc"][i], step)

    if figures:
        fig = fig_confusion(e["cm"], classes, step)
        writer.add_figure("Confusion_matrix", fig, step)
        plt.close(fig)
        fig = fig_roc(recorder.last["roc"], classes, step)
        writer.add_figure("ROC_Curve", fig, step)
        plt.close(fig)
        for pname, param in model.named_parameters():
            writer.add_histogram(f"Weights/{pname}", param.detach().cpu(), step)
    writer.flush()   # để TensorBoard thấy dữ liệu ngay khi đang train


def experiment_parameter(config, train_data, val_data):
    torch.manual_seed(Seed)
    print(f"Running experiment: {config['name']}")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
                              generator=torch.Generator().manual_seed(1))
    val_loader = DataLoader(val_data, num_workers=NUM_WORKERS, batch_size=256, shuffle=False)

    name_model = config["name"]
    model = MLP(hidden=config["hidden"], activation=config['activation'],
                normalization=config['normalization'], dropout=config['dropout'],
                residual_blocks=config['residual_blocks']).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # config không có label_smoothing -> mặc định 0.0 (trước đây gây KeyError)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    eval_loss_fn = nn.CrossEntropyLoss()
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                 if config["scheduler"] == "cosine" else None)
    model_dir = CHECKPOINT_DIR / name_model
    model_dir.mkdir(parents=True, exist_ok=True)

    last_path = model_dir / "last_checkpoint.pt"
    best_path = model_dir / "checkpoint_best.pt"
    best_acc, best_metrics = 0.0, None
    start_epoch = 0
    best_loss, bad, stopped = float("inf"), 0, False
    if last_path.exists():
        checkpoint = load_checkpoint(model, optimizer, last_path, scheduler)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        best_metrics = checkpoint["best_metrics"]
        best_loss = checkpoint.get("best_val_loss", float("inf"))
        bad = checkpoint.get("bad_epochs", 0)
        stopped = checkpoint.get("early_stopped", False)
    print(f"{name_model}: epoch {start_epoch}")

    recorder = RunRecorder(name_model, config, classes, REPORT_DIR,
                           n_params=n_params, device=device, resume_from=start_epoch)

    with SummaryWriter(str(LOG_DIR / name_model), flush_secs=10) as writer:
        writer.add_custom_scalars(tensorboard_layout())
        if start_epoch == 0:
            writer.add_text("Config", str(config), 0)
            writer.add_scalar("Model/parameters", n_params, 0)

        for epoch in range(start_epoch, EPOCHS):
            if stopped:
                print(f"{name_model}: đã early stop ở lần chạy trước")
                break
            torch.manual_seed(Seed + epoch)
            if device == "cuda":
                torch.cuda.manual_seed_all(Seed + epoch)

            lr = optimizer.param_groups[0]["lr"]            # LR dùng trong epoch này
            t0 = time.time()
            tr = train(model, train_loader, optimizer, loss_fn, epoch, writer, recorder)
            t_train = time.time() - t0
            t1 = time.time()
            val, y_true, y_prob = validate(model, val_loader, eval_loss_fn)
            t_val = time.time() - t1

            # early stopping + epoch tốt nhất (quyết định TRƯỚC khi ghi log để epoch cuối luôn được ghi)
            if val["loss"] < best_loss - 1e-4:
                best_loss, bad = val["loss"], 0
            else:
                bad += 1
            stopped = bad >= PATIENCE
            is_best = val["accuracy"] > best_acc
            if is_best:
                best_acc, best_metrics = val["accuracy"], dict(val)

            wnorm = weight_norm(model)
            e = recorder.log_epoch(epoch, tr, val, y_true, y_prob, lr=lr, weight_norm=wnorm,
                                   train_time=t_train, val_time=t_val, is_best=is_best)
            last_epoch = (epoch + 1 == EPOCHS) or stopped
            figures = (epoch == 0) or ((epoch + 1) % FIG_EVERY == 0) or last_epoch
            log_tensorboard(writer, model, epoch + 1, tr, val, e, lr, wnorm,
                            t_train, t_val, figures, recorder)

            if scheduler:
                scheduler.step()

            extra = {"best_val_loss": best_loss, "bad_epochs": bad, "early_stopped": stopped}
            if is_best:
                save_checkpoint(model, optimizer, epoch, best_acc, best_path, config, best_metrics,
                                scheduler, extra)
            save_checkpoint(model, optimizer, epoch, best_acc, last_path, config, best_metrics,
                            scheduler, extra)
            recorder.save()          # history.json được ghi sau mỗi epoch -> không mất dữ liệu nếu bị ngắt

            if stopped:
                print(f"Early stop tại epoch {epoch + 1}")
                break

    if best_metrics is None:
        raise RuntimeError(f"{name_model}: chưa có kết quả validation nào")
    recorder.finish(early_stopped=stopped)    # ghi reports/<tên model>/report.html
    return {"name": name_model, **best_metrics}


def compare_models(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    keys = ("accuracy", "precision", "recall", "f1", "fpr", "auc")
    names = [result["name"] for result in results]
    n_models = len(names)

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
                f"Comparison of {n_models} MLP Models - {key.upper()}",
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
        ax.set_title(f"Best Validation Metrics - All {n_models} MLP Models", fontsize=15, pad=12)

        for i in range(len(names)):
            for j in range(len(keys)):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        color="white" if colors[i, j] > 0.55 else "black", fontsize=10)

        fig.colorbar(image, ax=ax, shrink=0.8, label="Relative performance (higher is better)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "comparison_heatmap.png", dpi=200, bbox_inches="tight")
        writer.add_figure("Comparison/All_metrics_heatmap", fig, 0)
        plt.close(fig)

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
    print(f"{device}")
    # Fashion-MNIST: mean = 0.2860, std = 0.3530 (bản cũ gõ "0,286" nên Normalize bị lỗi std = 0)
    train_transform = Compose([RandomHorizontalFlip(), ToTensor(), Normalize((0.2860,), (0.3530,))])
    eval_transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
    train_data = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
    eval_data = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=eval_transform)
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(train_data), generator=g).tolist()
    train_data = Subset(train_data, perm[:50000])
    evaluate_data = Subset(eval_data, perm[50000:])

    print(f"Số lượng ảnh train: {len(train_data)}, validation: {len(evaluate_data)}")

    results = [experiment_parameter(cfg, train_data, evaluate_data)
               for cfg in Config_parameter]

    best = compare_models(results)
    source = CHECKPOINT_DIR / best["name"] / "checkpoint_best.pt"
    destination = CHECKPOINT_DIR / "checkpoint_best_global.pt"
    shutil.copy2(source, destination)

    # Báo cáo HTML tổng hợp tất cả mô hình -> reports/report_all_models.html
    build_report(load_runs(REPORT_DIR, [cfg["name"] for cfg in Config_parameter]),
                 REPORT_DIR / "report_all_models.html",
                 title="Báo cáo huấn luyện - so sánh các mô hình MLP (Fashion-MNIST)")
