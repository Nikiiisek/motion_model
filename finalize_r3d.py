from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from motion_models.data_utils.dataset_fixedlen import FixedLenVideoDataset
from motion_models.data_utils.transforms import get_r3d_val_transforms
from motion_models.models.R3D18 import R3D18Classifier


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels, video_ids in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    data_root = Path(
        "/home/nikol/Documents/dataset_private/processed_16f_subject_v2"
    )

    report_dir = Path(
        "/home/nikol/Documents/motion_model/"
        "motion_models/reports/"
        "split_subject_v2/r3d18/baseline_10ep"
    )

    checkpoint_path = report_dir / "best.pth"

    class_names = ["T", "pase"]


    test_dataset = FixedLenVideoDataset(
        root_dir=data_root / "test",
        class_names=class_names,
        transform=get_r3d_val_transforms(),
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


    model = R3D18Classifier(
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )

    criterion = torch.nn.CrossEntropyLoss()

    print("Best checkpoint loaded:", checkpoint_path)

    test_loss, test_acc = validate_one_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    all_labels = []
    all_preds = []
    wrong = []

    model.eval()

    with torch.no_grad():
        for videos, labels, video_ids in test_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

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

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(class_report)

    print(f"\nWrong predictions: {len(wrong)}")


    (
        report_dir / "classification_report.txt"
    ).write_text(
        class_report,
        encoding="utf-8",
    )


    wrong_lines = ["video_id,true,pred"]

    for item in wrong:
        wrong_lines.append(
            f"{item['video_id']},"
            f"{item['true']},"
            f"{item['pred']}"
        )

    (
        report_dir / "wrong_predictions.csv"
    ).write_text(
        "\n".join(wrong_lines) + "\n",
        encoding="utf-8",
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=class_names,
    )

    display.plot()

    plt.title("R3D-18 - Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        report_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    config = {
        "dataset_version": "split_subject_v2",
        "dataset_path": str(data_root),
        "split_type": "subject_independent",
        "model": "R3D-18",
        "pretrained": True,
        "pretraining_dataset": "Kinetics-400",
        "classes": class_names,
        "num_classes": 2,
        "num_frames": 16,
        "input_size": "112x112",
        "augmentation": False,
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "train_samples": 631,
        "val_samples": 135,
        "test_samples": 135,
        "best_epoch": 3,
        "best_val_loss": 0.0170,
        "best_val_acc": 1.0000,
    }

    with open(
        report_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    history = [
        [1, 0.000100, 0.4451, 0.7924, 0.0972, 0.9852],
        [2, 0.000100, 0.1072, 0.9762, 0.2851, 0.9333],
        [3, 0.000100, 0.0404, 0.9968, 0.0170, 1.0000],
        [4, 0.000100, 0.0619, 0.9826, 0.1665, 0.9333],
        [5, 0.000100, 0.0360, 0.9937, 0.0502, 0.9926],
        [6, 0.000100, 0.0551, 0.9857, 0.0354, 0.9926],
        [7, 0.000050, 0.0105, 0.9984, 0.1559, 0.9556],
        [8, 0.000050, 0.0057, 1.0000, 0.0946, 0.9704],
        [9, 0.000050, 0.0036, 1.0000, 0.0847, 0.9704],
        [10, 0.000025, 0.0029, 1.0000, 0.0625, 0.9852],
    ]

    metrics_lines = [
        "epoch,learning_rate,train_loss,train_acc,val_loss,val_acc"
    ]

    for row in history:
        metrics_lines.append(
            ",".join(str(value) for value in row)
        )

    (
        report_dir / "metrics.csv"
    ).write_text(
        "\n".join(metrics_lines) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # RESULTS TXT
    # ---------------------------------------------------------

    results = (
        "Dataset: split_subject_v2\n"
        "Model: R3D-18\n"
        "Pretraining: Kinetics-400\n"
        "Experiment: baseline_10ep\n"
        "Augmentation: none\n"
        "Input size: 112x112\n"
        "Frames per video: 16\n"
        "Epochs: 10\n"
        "Learning rate: 0.0001\n"
        "Batch size: 2\n"
        "\n"
        "Train samples: 631\n"
        "Val samples: 135\n"
        "Test samples: 135\n"
        "\n"
        "Best Epoch: 3\n"
        "Best Val Loss: 0.0170\n"
        "Val Acc at Best Epoch: 1.0000\n"
        "\n"
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Acc: {test_acc:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        "\n"
        "Confusion Matrix:\n"
        f"{conf_matrix}\n"
        "\n"
        "Classification Report:\n"
        f"{class_report}\n"
        f"Wrong predictions: {len(wrong)}\n"
    )

    (
        report_dir / "results.txt"
    ).write_text(
        results,
        encoding="utf-8",
    )

    print("\nReports saved to:")
    print(report_dir)


if __name__ == "__main__":
    main()