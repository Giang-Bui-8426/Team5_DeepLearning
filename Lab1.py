import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.tensorboard import SummaryWriter

DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"

# Đổi tên thư mục này để tạo một "run" mới trên TensorBoard
LOG_DIR = "./runs/fashionmnist_experiment"

os.makedirs(CKPT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang dùng thiết bị: {device}")

transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"Số lượng ảnh train: {len(train_data)}, test: {len(test_data)}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(LOG_DIR)


def train_one_epoch(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Loss/train_batch", loss.item(), step)
            print(f"  [Epoch {epoch+1}] batch {batch_idx}/{len(dataloader)} - loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct = 0.0, 0
    num_samples = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / num_samples
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  -> Đã lưu checkpoint tại: {path}")


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"  -> Đã load checkpoint từ: {path} (tiếp tục từ epoch {start_epoch})")
    return start_epoch


if __name__ == "__main__":
    best_acc = 0.0
    start_epoch = 0

    last_ckpt = os.path.join(CKPT_DIR, "last_checkpoint.pt")
    if os.path.exists(last_ckpt):
        start_epoch = load_checkpoint(model, optimizer, last_ckpt)

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, epoch)
        test_loss, test_acc = evaluate(test_loader, model, loss_fn)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/test_epoch", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        save_checkpoint(model, optimizer, epoch, last_ckpt)

        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(CKPT_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Model tốt nhất mới (acc={best_acc*100:.2f}%), đã lưu: {best_path}")

    writer.close()
    print(f"\nHoàn tất huấn luyện. Accuracy tốt nhất trên test: {best_acc*100:.2f}%")

    inference_model = NeuralNetwork().to(device)
    inference_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pt"), map_location=device))
    inference_model.eval()

    X_sample, y_sample = test_data[0]
    X_sample = X_sample.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = inference_model(X_sample)
        predicted_class = classes[pred.argmax(1).item()]
        actual_class = classes[y_sample]
    print(f"\nVí dụ dự đoán: Predicted = '{predicted_class}', Actual = '{actual_class}'")

    # Mở TensorBoard để xem biểu đồ loss/accuracy:
    # tensorboard --logdir=./runs/fashionmnist_experiment