import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_utils.dataset_varlen import VarLenVideoDataset
from data_utils.collate_varlen import pad_collate_varlen
from Cnn_LSTM import CnnLstmClassifier

def get_data_root():
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        raise RuntimeError(
            "DATA_ROOT is not set. Set it as environment variable, e.g.\n"
            "export DATA_ROOT=/path/to/your/private/dataset/train"
        )
    if not os.path.exists(data_root):
        raise RuntimeError(f"DATA_ROOT path does not exist: {data_root}")
    return data_root

def main():
    data_root = get_data_root()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = VarLenVideoDataset(data_root, transform=transform)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=pad_collate_varlen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    num_classes = len(ds.classes)
    model = CnnLstmClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    max_steps = 5
    for step, (xb, yb, lengths) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb, lengths)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        print(f"step {step} loss {loss.item():.4f} preds {preds.tolist()} labels {yb.tolist()} lengths {lengths.tolist()}")

        if step + 1 >= max_steps:
            break

    print("classes:", ds.classes)

if __name__ == "__main__":
    main()

