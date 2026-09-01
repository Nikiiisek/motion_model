from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from motion_models.data_utils.config import PROCESSED_DIR
from motion_models.data_utils.dataset_fixedlen import FixedLenVideoDataset
from motion_models.data_utils.transforms import get_train_transforms, get_val_transforms
from motion_models.models.mobilenet_lstm import MobileNetV3SmallLSTM
from motion_models.data_utils.seed import set_seed
from motion_models.utils.experiment_logger import save_experiment_to_csv


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

@torch.no_grad()
def list_wrong_predictions(model, loader, device, class_names):
    model.eval()

    wrong = []

    for videos, labels, video_ids in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        preds = outputs.argmax(dim=1)

        for i in range(len(video_ids)):
            true_label = labels[i].item()
            pred_label = preds[i].item()

            if true_label != pred_label:
                wrong.append({
                    "video_id": video_ids[i],
                    "true": class_names[true_label],
                    "pred": class_names[pred_label],
                })

    print(f"\nWrong predictions: {len(wrong)}")

    for item in wrong:
        print(item)

    return wrong

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = PROCESSED_DIR

    print("data_root:", data_root)
    print("train exists:", (data_root / "train").exists())
    print("val exists:", (data_root / "val").exists())
    print("test exists:", (data_root / "test").exists())

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    class_names = ["T", "pase"]
    num_classes = len(class_names)

    batch_size = 4
    num_workers = 0
    num_epochs = 10
    learning_rate = 5e-4

    train_dataset = FixedLenVideoDataset(
        root_dir=train_dir,
        class_names=class_names,
        transform=get_train_transforms(),
        augment=True,
    )

    val_dataset = FixedLenVideoDataset(
        root_dir=val_dir,
        class_names=class_names,
        transform=get_val_transforms(),
        augment=False,
    )

    test_dataset = FixedLenVideoDataset(
        root_dir=test_dir,
        class_names=class_names,
        transform=get_val_transforms(),
        augment=False,
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

    for videos, labels, video_ids in train_loader:
        print("videos shape:", videos.shape)
        print("labels shape:", labels.shape)
        print("example video_ids:", video_ids[:2])
        break

    model = MobileNetV3SmallLSTM(
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
    best_val_acc = 0.0
    best_epoch = 0

    dataset_version = "dataset_v2"
    experiment_name = "color_aug_10ep"

    project_root = Path(__file__).resolve().parents[1]

    report_dir = (
        project_root
        / "reports"
        / dataset_version
        / "mobilenet_lstm"
        / experiment_name
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    save_path = report_dir / "best.pth"
    log_path = report_dir / "results.txt"


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
            best_val_acc = val_acc
            best_epoch = epoch + 1

            torch.save(model.state_dict(), save_path)

            print(
                f"Best model saved to: {save_path} "
                f"(epoch {best_epoch}, val_loss={best_val_loss:.4f}, "
                f"val_acc={best_val_acc:.4f})"
            )

    print("Training finished.")

    # načtení nejlepšího modelu a finální validace/test
    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True
        )
    )


    test_loss, test_acc = validate_one_epoch(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for videos, labels, video_ids in test_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    with open(log_path, "w") as f:
        f.write(f"Dataset version: {dataset_version}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write("Model: MobileNetV3Small + LSTM\n")
        f.write("Augmentation: none\n")
        f.write("Input size: 112x112\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("Seed: 42\n")

        f.write("\nDataset sizes:\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")

        f.write("\n")
        f.write(f"Final Train Loss: {train_loss:.4f}\n")
        f.write(f"Final Train Acc: {train_acc:.4f}\n")
        f.write(f"Final Val Loss: {val_loss:.4f}\n")
        f.write(f"Final Val Acc: {val_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Val acc at Best Epoch: {best_val_acc:.4f}\n")

        f.write("\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")

        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(all_labels, all_preds)))

        f.write("\n\nClassification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    wrong = list_wrong_predictions(
        model,
        test_loader,
        device,
        class_names
    )

    # Classification report také jako slovník kvůli uložení F1 do CSV
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]


# Cesta k centrálnímu CSV se všemi experimenty
    csv_path = project_root / "reports" / "experiments.csv"


    experiment_data = {
        "dataset": dataset_version,
        "model": "MobileNetV3Small_LSTM",
        "experiment": experiment_name,
        "augmentation": "brightness_contrast",
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": 42,

        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),

        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 4),

        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


    save_experiment_to_csv(
        csv_path,
        experiment_data
    )

    print(f"Experiment added to CSV: {csv_path}")

if __name__ == "__main__":
    main()