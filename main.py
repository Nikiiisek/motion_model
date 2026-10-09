from pathlib import Path

import torch
from torch.utils.data import DataLoader

from motion_models.data_utils.dataset_fixedlen import FixedLenVideoDataset
from motion_models.data_utils.transforms import get_r3d_val_transforms
from motion_models.models.R3D18 import R3D18Classifier


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    data_root = Path(
        "/home/nikol/Documents/dataset_private/processed_16f_subject_v2"
    )

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
        num_workers=0,
    )

    model = R3D18Classifier(
        num_classes=len(class_names),
        pretrained=True,
    ).to(device)

    model.eval()

    with torch.no_grad():

        for videos, labels, video_ids in test_loader:

            print("Dataset output:", videos.shape)

            videos = videos.to(device)

            outputs = model(videos)

            print("Model output:", outputs.shape)
            print("Labels:", labels)
            print("Video IDs:", video_ids)

            break


if __name__ == "__main__":
    main()