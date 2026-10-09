from pathlib import Path
from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from motion_models.models.R3D18 import R3D18Classifier
from motion_models.data_utils.dataset_fixedlen import FixedLenVideoDataset
from motion_models.data_utils.transforms import (
    get_r3d_train_transforms,
    get_r3d_val_transforms,
)
from motion_models.data_utils.seed import set_seed
from sklearn.metrics import confusion_matrix, classification_report

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels, video_ids in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)
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

    seed = 42
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data_root = Path(
        "/home/nikol/Documents/dataset_private/processed_16f_subject_v2"
    )

    class_names = ["T", "pase"]
    num_classes = len(class_names)

    batch_size = 2
    num_workers = 4
    num_epochs = 10
    learning_rate = 1e-4

    dataset_version = "split_subject_v2"
    experiment_name = "baseline_10ep"
    augmentation_name = "none"

    report_root = Path(
        "/home/nikol/Documents/motion_model/motion_models/reports"
    )

    report_dir = (
        report_root
        / dataset_version
        / "r3d18"
        / experiment_name
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = report_dir / "best.pth"

    train_dataset = FixedLenVideoDataset(
        root_dir=data_root / "train",
        class_names=class_names,
        transform=get_r3d_train_transforms(),
        augment=False,
    )

    val_dataset = FixedLenVideoDataset(
        root_dir=data_root / "val",
        class_names=class_names,
        transform=get_r3d_val_transforms(),
        augment=False,
    )

    test_dataset = FixedLenVideoDataset(
        root_dir=data_root / "test",
        class_names=class_names,
        transform=get_r3d_val_transforms(),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    for videos, labels, video_ids in train_loader:
        print("videos shape:", videos.shape)
        print("labels shape:", labels.shape)
        print("example video_ids:", video_ids[:2])
        break

    model = R3D18Classifier(
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print("R3D-18 model loaded.")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )   

    CHECK_ONLY = False

    if CHECK_ONLY:
        print(
            "CHECK_ONLY=True -> kontrola dokončena, "
            "trénink se nespouští."
        )
        return
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(
                f"Best model saved to: {save_path} "
                f"(epoch {best_epoch}, "
                f"val_loss={best_val_loss:.4f}, "
                f"val_acc={best_val_acc:.4f})"
            )

        scheduler.step(val_loss)

    print("Training finished.")

    # ---------------------------------------------------------
# TEST NEJLEPSIHO CHECKPOINTU
# ---------------------------------------------------------

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True,
        )
    )

    test_loss, test_acc = validate_one_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )

    all_preds = []
    all_labels = []
    wrong = []

    model.eval()

    with torch.no_grad():
        for videos, labels, video_ids in test_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i in range(len(video_ids)):

                true_label = labels[i].item()
                pred_label = preds[i].item()

                if true_label != pred_label:
                    wrong.append({
                        "video_id": video_ids[i],
                        "true": class_names[true_label],
                        "pred": class_names[pred_label],
                    })


    conf_matrix = confusion_matrix(
        all_labels,
        all_preds,
    )

    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )

    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]


    print("Confusion Matrix:")
    print(conf_matrix)

    print("Classification Report:")
    print(class_report)

    print(f"\nWrong predictions: {len(wrong)}")

    for item in wrong:
        print(item)

if __name__ == "__main__":
    main()