from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from motion_models.data_utils.dataset_fixedlen import FixedLenVideoDataset
from motion_models.data_utils.transforms import get_val_transforms
from motion_models.models.mobilenet_lstm import MobileNetV3SmallLSTM
from motion_models.data_utils.seed import set_seed


# --------------------------------------------------
# NASTAVENÍ
# --------------------------------------------------

set_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


DATA_ROOT = Path(
    "/home/nikol/Documents/dataset_private/processed_16f_subject_v2"
)

REPORT_DIR = Path(
    "/home/nikol/Documents/motion_model/"
    "motion_models/reports/"
    "split_subject_v2/"
    "mobilenetv3_lstm/"
    "baseline_10ep"
)

MODEL_PATH = REPORT_DIR / "best.pth"
METRICS_PATH = REPORT_DIR / "metrics.csv"

RESULTS_PATH = REPORT_DIR / "results.txt"
CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR / "classification_report.txt"
)
CONFUSION_MATRIX_PATH = (
    REPORT_DIR / "confusion_matrix.png"
)
WRONG_PREDICTIONS_PATH = (
    REPORT_DIR / "wrong_predictions.csv"
)


class_names = ["T", "pase"]

batch_size = 4
num_workers = 0


# --------------------------------------------------
# TEST DATASET
# --------------------------------------------------

test_dir = DATA_ROOT / "test"

test_dataset = FixedLenVideoDataset(
    root_dir=test_dir,
    class_names=class_names,
    transform=get_val_transforms(),
    augment=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

print("Test samples:", len(test_dataset))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = MobileNetV3SmallLSTM(
    num_classes=len(class_names),

    # Pretrained zde nepotřebujeme,
    # protože načítáme celý uložený state_dict.
    pretrained=False,
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

print("Model loaded from:")
print(MODEL_PATH)


criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# TEST
# --------------------------------------------------

model.eval()

running_loss = 0.0
correct = 0
total = 0

all_labels = []
all_preds = []
wrong_predictions = []


with torch.no_grad():

    for videos, labels, video_ids in test_loader:

        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        loss = criterion(
            outputs,
            labels
        )

        preds = outputs.argmax(dim=1)

        running_loss += (
            loss.item() * videos.size(0)
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_preds.extend(
            preds.cpu().tolist()
        )

        for i in range(len(video_ids)):

            true_label = labels[i].item()
            pred_label = preds[i].item()

            if true_label != pred_label:

                wrong_predictions.append({
                    "video_id": video_ids[i],
                    "true": class_names[true_label],
                    "pred": class_names[pred_label],
                })


test_loss = running_loss / total
test_acc = correct / total


print()
print("----------------------------------")
print("TEST RESULTS")
print("----------------------------------")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Acc:  {test_acc:.4f}"
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_preds
)

print()
print("Confusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "MobileNetV3 Small + LSTM\n"
    "Subject-independent test"
)

fig.tight_layout()

fig.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

classification_report_text = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4,
)

print()
print("Classification Report:")
print(classification_report_text)


with open(
    CLASSIFICATION_REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        classification_report_text
    )


# --------------------------------------------------
# WRONG PREDICTIONS
# --------------------------------------------------

with open(
    WRONG_PREDICTIONS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id",
            "true",
            "pred",
        ],
    )

    writer.writeheader()
    writer.writerows(
        wrong_predictions
    )


# --------------------------------------------------
# NAČTENÍ NEJLEPŠÍ EPOCHY Z METRICS.CSV
# --------------------------------------------------

best_epoch = None
best_val_loss = None
best_val_acc = None

with open(
    METRICS_PATH,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)

    if rows:

        best_row = min(
            rows,
            key=lambda row: float(
                row["val_loss"]
            )
        )

        best_epoch = int(
            best_row["epoch"]
        )

        best_val_loss = float(
            best_row["val_loss"]
        )

        best_val_acc = float(
            best_row["val_acc"]
        )


# --------------------------------------------------
# RESULTS.TXT
# --------------------------------------------------

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Dataset: split_subject_v2\n"
    )

    f.write(
        "Split type: subject-independent\n"
    )

    f.write(
        "Model: MobileNetV3Small + LSTM\n"
    )

    f.write(
        "Experiment: baseline_10ep\n"
    )

    f.write(
        "Frames per video: 16\n"
    )

    f.write(
        f"Test samples: {len(test_dataset)}\n"
    )

    f.write("\n")

    f.write(
        f"Best Epoch: {best_epoch}\n"
    )

    f.write(
        f"Best Val Loss: {best_val_loss:.4f}\n"
    )

    f.write(
        f"Val Acc at Best Epoch: "
        f"{best_val_acc:.4f}\n"
    )

    f.write("\n")

    f.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    f.write(
        f"Test Acc: {test_acc:.4f}\n"
    )

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        classification_report_text
    )


print()
print(
    "Wrong predictions:",
    len(wrong_predictions)
)

print()
print("Evaluation finished.")
print("Results saved to:")
print(REPORT_DIR)