"""
LAB 1 - FashionMNIST Classification với MLP
--------------------------------------------
Nhiệm vụ:
  1. Tạo hàm evaluate (tham khảo Chapter 2 - slide 14: Accuracy, Precision, Recall,
     F1-Score, Confusion Matrix; Chapter 3 - slide 10-11: model.eval(), torch.no_grad()).
  2. Huấn luyện với 3 cấu hình hyperparameter khác nhau (Chapter 3: learning rate,
     batch size, optimizer, dropout, L2 weight_decay, BatchNorm, weight init) và so sánh.
  3. Hiển thị một vài ảnh predicted vs actual.

Chạy:  python lab1_train_evaluate.py
Yêu cầu: pip install torch torchvision matplotlib
"""

import copy
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ======================================================================
# 0. Thiết lập chung
# ======================================================================
SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chapter 3 - slide 19
print("Device:", device)

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
NUM_CLASSES = len(CLASSES)


# ======================================================================
# 1. Chuẩn bị dữ liệu (Chapter 3 - slide 4, 5)
# ======================================================================
transform = transforms.Compose([
    transforms.ToTensor(),                       # PIL -> Tensor, giá trị [0, 1]
    transforms.Normalize((0.2860,), (0.3530,)),  # mean/std của FashionMNIST
])

full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Tách 60k ảnh train thành 50k train + 10k validation (Chapter 3 - slide 10:
# validation là tập riêng để theo dõi overfitting và chọn model tốt nhất)
train_dataset, val_dataset = random_split(
    full_train, [50000, 10000], generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# ======================================================================
# 2. Định nghĩa model - MLP từ lab1.py, mở rộng thêm tham số để thử hyperparameter
# ======================================================================
class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size1=128, hidden_size2=64,
                 num_classes=10, dropout=0.2, use_batchnorm=False):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        # Tầng ẩn 1: 784 -> hidden_size1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1) if use_batchnorm else nn.Identity()  # Chapter 3 - slide 18
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)                                      # Chapter 3 - slide 22

        # Tầng ẩn 2: hidden_size1 -> hidden_size2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        # Output: trả về logits (CrossEntropyLoss đã gồm Softmax)
        self.out = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.out(x)


def init_weights_he(m):
    """He (Kaiming) initialization - tối ưu cho ReLU (Chapter 3 - slide 17)."""
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity="relu")
        init.zeros_(m.bias)


# ======================================================================
# 3. HÀM EVALUATE (Chapter 2 - slide 14 + Chapter 3 - slide 10, 11)
# ======================================================================
def evaluate(model, dataloader, criterion, device, num_classes=NUM_CLASSES):
    """
    Đánh giá model trên một tập dữ liệu (validation hoặc test).

    Trả về dict gồm:
      - loss       : loss trung bình
      - accuracy   : % dự đoán đúng
      - precision  : Precision trung bình (macro) = TP / (TP + FP)
      - recall     : Recall trung bình (macro)    = TP / (TP + FN)
      - f1         : F1 trung bình (macro)        = 2PR / (P + R)
      - per_class  : precision/recall/f1 cho từng lớp
      - confusion  : Confusion matrix [num_classes x num_classes]
                     (hàng = nhãn thật, cột = nhãn dự đoán)
    """
    was_training = model.training     # nhớ chế độ hiện tại để khôi phục sau khi đánh giá
    model.eval()                      # tắt Dropout, BatchNorm dùng running stats
    total_loss, total_samples = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():             # không tính gradient -> nhanh, ít bộ nhớ
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            preds = outputs.argmax(dim=1)
            # Cộng dồn vào confusion matrix
            idx = labels.cpu() * num_classes + preds.cpu()
            confusion += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)

    cm = confusion.float()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp           # dự đoán là lớp c nhưng sai
    fn = cm.sum(dim=1) - tp           # thật là lớp c nhưng đoán sai

    precision_c = tp / (tp + fp).clamp(min=1)
    recall_c = tp / (tp + fn).clamp(min=1)
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c).clamp(min=1e-12)

    model.train(was_training)         # trả model về chế độ cũ (Chapter 3 - slide 10)

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * tp.sum().item() / total_samples,
        "precision": 100.0 * precision_c.mean().item(),
        "recall": 100.0 * recall_c.mean().item(),
        "f1": 100.0 * f1_c.mean().item(),
        "per_class": {
            "precision": precision_c.tolist(),
            "recall": recall_c.tolist(),
            "f1": f1_c.tolist(),
        },
        "confusion": confusion,
    }


# ======================================================================
# 4. Training loop (Chapter 3 - slide 8, 9)
# ======================================================================
def train_one_config(cfg):
    print("\n" + "=" * 70)
    print(f"CẤU HÌNH: {cfg['name']}")
    print("  " + ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "name"))
    print("=" * 70)

    torch.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = MLP(hidden_size1=cfg["hidden1"], hidden_size2=cfg["hidden2"],
                dropout=cfg["dropout"], use_batchnorm=cfg["batchnorm"]).to(device)
    if cfg["he_init"]:
        model.apply(init_weights_he)

    criterion = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                              weight_decay=cfg["weight_decay"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None
    start = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)   # 1. Move to device
            optimizer.zero_grad()                               # 2. Zero gradients
            outputs = model(data)                               # 3. Forward
            loss = criterion(outputs, labels)                   # 4. Loss
            loss.backward()                                     # 5. Backward
            optimizer.step()                                    # 6. Update weights

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        train_loss, train_acc = running_loss / seen, 100.0 * correct / seen
        val = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])

        # Giữ lại model tốt nhất theo validation accuracy (Chapter 3 - slide 10)
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch + 1:2d}/{cfg['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val['loss']:.4f} | Val Acc: {val['accuracy']:.2f}%")

    # Lưu và load lại model tốt nhất (Chapter 3 - slide 12)
    path = f"model_{cfg['name'].split()[0]}.pth"
    torch.save(best_state, path)
    best_model = MLP(hidden_size1=cfg["hidden1"], hidden_size2=cfg["hidden2"],
                     dropout=cfg["dropout"], use_batchnorm=cfg["batchnorm"]).to(device)
    best_model.load_state_dict(torch.load(path, map_location=device))
    best_model.eval()

    test = evaluate(best_model, test_loader, criterion, device)
    print(f"--> TEST (best model): Acc={test['accuracy']:.2f}% | Precision={test['precision']:.2f}% | "
          f"Recall={test['recall']:.2f}% | F1={test['f1']:.2f}% | thời gian {time.time() - start:.1f}s")

    return {"cfg": cfg, "history": history, "test": test, "model": best_model}


# ======================================================================
# 5. Ba cấu hình hyperparameter (Chapter 3 - slide 16-22)
# ======================================================================
EPOCHS = 10
CONFIGS = [
    {   # Cấu hình gốc của lab1.py, Adam là "good default" (Chapter 3 - slide 7)
        "name": "C1 Baseline (Adam, lr=1e-3, bs=64, dropout=0.2)",
        "optimizer": "adam", "lr": 1e-3, "batch_size": 64, "epochs": EPOCHS,
        "hidden1": 128, "hidden2": 64, "dropout": 0.2, "weight_decay": 0.0,
        "batchnorm": False, "he_init": False,
    },
    {   # Đổi optimizer + learning rate lớn hơn + batch size lớn hơn
        "name": "C2 SGD-Momentum (lr=0.05, bs=256, dropout=0.2)",
        "optimizer": "sgd", "lr": 0.05, "batch_size": 256, "epochs": EPOCHS,
        "hidden1": 128, "hidden2": 64, "dropout": 0.2, "weight_decay": 0.0,
        "batchnorm": False, "he_init": False,
    },
    {   # Mạng rộng hơn + các kỹ thuật chống overfitting / vanishing gradient
        "name": "C3 Wide+Regularized (Adam, 512-256, BN, He init, dropout=0.5, L2=1e-4)",
        "optimizer": "adam", "lr": 1e-3, "batch_size": 64, "epochs": EPOCHS,
        "hidden1": 512, "hidden2": 256, "dropout": 0.5, "weight_decay": 1e-4,
        "batchnorm": True, "he_init": True,
    },
]


# ======================================================================
# 6. Trực quan hóa
# ======================================================================
def plot_comparison(results, filename="compare_configs.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for r in results:
        label = r["cfg"]["name"].split(" (")[0]
        ep = range(1, len(r["history"]["train_loss"]) + 1)
        axes[0].plot(ep, r["history"]["train_loss"], "-o", ms=3, label=f"{label} - train")
        axes[0].plot(ep, r["history"]["val_loss"], "--", label=f"{label} - val")
        axes[1].plot(ep, r["history"]["val_acc"], "-o", ms=3, label=label)

    axes[0].set(title="Loss theo epoch", xlabel="Epoch", ylabel="CrossEntropy Loss")
    axes[1].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)

    # Biểu đồ cột so sánh metric trên tập test
    metrics = ["accuracy", "precision", "recall", "f1"]
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        vals = [r["test"][m] for m in metrics]
        xs = [j + i * width for j in range(len(metrics))]
        bars = axes[2].bar(xs, vals, width, label=r["cfg"]["name"].split(" (")[0])
        for b, v in zip(bars, vals):
            axes[2].text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7)
    axes[2].set_xticks([j + width * (len(results) - 1) / 2 for j in range(len(metrics))])
    axes[2].set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    lo = min(r["test"][m] for r in results for m in metrics)
    axes[2].set_ylim(lo - 3, 100)
    axes[2].set(title="Metric trên tập Test (%)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def plot_confusion(confusion, title, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    thresh = confusion.max().item() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = confusion[i, j].item()
            ax.text(j, i, v, ha="center", va="center", fontsize=7,
                    color="white" if v > thresh else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def show_predictions(model, dataset, n=16, filename="predicted_vs_actual.png"):
    """Hiển thị ảnh kèm nhãn Predicted vs Actual (xanh = đúng, đỏ = sai)."""
    model.eval()
    g = torch.Generator().manual_seed(7)
    idxs = torch.randperm(len(dataset), generator=g)[:n].tolist()
    images = torch.stack([dataset[i][0] for i in idxs])
    labels = torch.tensor([dataset[i][1] for i in idxs])

    with torch.no_grad():
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
    conf, preds = probs.max(dim=1)

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.4))
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= n:
            continue
        img = images[k].squeeze() * 0.3530 + 0.2860       # bỏ normalize để hiển thị
        ax.imshow(img, cmap="gray")
        ok = preds[k] == labels[k]
        ax.set_title(f"P: {CLASSES[preds[k]]} ({conf[k] * 100:.0f}%)\nA: {CLASSES[labels[k]]}",
                     fontsize=8, color="green" if ok else "red")
    plt.suptitle("Predicted (P) vs Actual (A) — xanh: đúng, đỏ: sai", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


def show_mistakes(model, dataloader, n=8, filename="misclassified.png"):
    """Hiển thị riêng các ảnh bị dự đoán sai để phân tích lỗi."""
    model.eval()
    wrong_imgs, wrong_pred, wrong_true = [], [], []
    with torch.no_grad():
        for data, labels in dataloader:
            preds = model(data.to(device)).argmax(1).cpu()
            mask = preds != labels
            wrong_imgs.append(data[mask]); wrong_pred.append(preds[mask]); wrong_true.append(labels[mask])
            if sum(len(w) for w in wrong_imgs) >= n:
                break
    imgs, p, t = torch.cat(wrong_imgs)[:n], torch.cat(wrong_pred)[:n], torch.cat(wrong_true)[:n]

    fig, axes = plt.subplots(1, n, figsize=(n * 1.9, 2.6))
    for k, ax in enumerate(axes):
        ax.imshow(imgs[k].squeeze() * 0.3530 + 0.2860, cmap="gray")
        ax.set_title(f"P: {CLASSES[p[k]]}\nA: {CLASSES[t[k]]}", fontsize=8, color="red")
        ax.axis("off")
    plt.suptitle("Một số ảnh bị dự đoán SAI", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


# ======================================================================
# 7. MAIN
# ======================================================================
if __name__ == "__main__":
    results = [train_one_config(cfg) for cfg in CONFIGS]

    # Bảng so sánh
    print("\n" + "=" * 100)
    print(f"{'Cấu hình':<45}{'Best Val Acc':>13}{'Test Acc':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Gap*':>7}")
    print("-" * 100)
    for r in results:
        h = r["history"]
        gap = h["train_acc"][-1] - h["val_acc"][-1]
        print(f"{r['cfg']['name'].split(' (')[0]:<45}{max(h['val_acc']):>12.2f}%"
              f"{r['test']['accuracy']:>9.2f}%{r['test']['precision']:>10.2f}%"
              f"{r['test']['recall']:>8.2f}%{r['test']['f1']:>7.2f}%{gap:>6.2f}")
    print("* Gap = Train Acc - Val Acc ở epoch cuối (càng lớn càng có dấu hiệu overfitting)")

    # Chọn cấu hình theo validation, tập test chỉ dùng để báo cáo (Chapter 3 - slide 10)
    best = max(results, key=lambda r: max(r["history"]["val_acc"]))
    print(f"\nCấu hình tốt nhất (theo Best Val Acc): {best['cfg']['name']}")

    # Per-class F1 của cấu hình tốt nhất
    print("\nF1 từng lớp (cấu hình tốt nhất):")
    for c, f1 in zip(CLASSES, best["test"]["per_class"]["f1"]):
        print(f"  {c:<12} {f1 * 100:6.2f}%")

    plot_comparison(results)
    plot_confusion(best["test"]["confusion"], f"Confusion Matrix - {best['cfg']['name'].split(' (')[0]}")
    show_predictions(best["model"], test_dataset, n=16)
    show_mistakes(best["model"], test_loader, n=8)
