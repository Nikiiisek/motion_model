from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from motion_models.data_utils.dataset_fixedlen import FixedLenVideoDataset
from motion_models.data_utils.transforms import get_train_transforms, get_val_transforms
from motion_models.models.mobilenet_small import MobileNetV3SmallBaseline
from motion_models.data_utils.seed import set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels, video_ids in loader:
        videos = videos.to(device)   # (B, T, C, H, W)
        labels = labels.to(device)   # (B,)

        optimizer.zero_grad()

        outputs = model(videos)      # (B, num_classes)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels, video_ids in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * videos.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path("/Users/nikolpavlovska/PycharmProjects/dataset_private/processed_16f")

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    class_names = ["T", "pase"]
    num_classes = len(class_names)

    batch_size = 4
    num_workers = 0
    num_epochs = 15
    learning_rate = 1e-3

    train_dataset = FixedLenVideoDataset(
        root_dir=train_dir,
        class_names=class_names,
        transform=get_train_transforms()
    )

    val_dataset = FixedLenVideoDataset(
        root_dir=val_dir,
        class_names=class_names,
        transform=get_val_transforms()
    )

    test_dataset = FixedLenVideoDataset(
        root_dir=test_dir,
        class_names=class_names,
        transform=get_val_transforms()
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # debug tvaru batch
    for videos, labels, video_ids in train_loader:
        print("videos shape:", videos.shape)
        print("labels shape:", labels.shape)
        print("example video_ids:", video_ids[:2])
        break

    model = MobileNetV3SmallBaseline(
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_loss = float("inf")
    save_path = Path("best_mobilenet_small_baseline.pth")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to: {save_path}")

    print("Training finished.")

    # načtení nejlepšího modelu a finální validace/test
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc = validate_one_epoch(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()