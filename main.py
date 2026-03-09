from pathlib import Path
from torchvision import transforms
from data_utils.dataset_fixedlen import FixedLenVideoDataset


def get_data_root():
    data_root = Path("/Users/nikolpavlovska/PycharmProjects/dataset_private/processed_16f")
    if not data_root.exists():
        raise RuntimeError(f"DATA_ROOT path does not exist: {data_root}")
    return data_root


def main():
    data_root = get_data_root()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = FixedLenVideoDataset(
        root_dir=data_root / "train",
        class_names=["pase", "T"],
        transform=transform
    )

    val_ds = FixedLenVideoDataset(
        root_dir=data_root / "val",
        class_names=["pase", "T"],
        transform=transform
    )

    test_ds = FixedLenVideoDataset(
        root_dir=data_root / "test",
        class_names=["pase", "T"],
        transform=transform
    )

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    x, y, vid = train_ds[0]
    print("Shape:", x.shape)
    print("Label:", y)
    print("Video ID:", vid)


if __name__ == "__main__":
    main()

