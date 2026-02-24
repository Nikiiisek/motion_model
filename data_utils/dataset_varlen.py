import os
import re
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def parse_prefix_and_frame(fname: str):
    """
    Expects: PL2_frame_0003.jpg
    returns: ("PL2", 3)
    """
    base = os.path.basename(fname)
    prefix = base.split("_")[0]  # PL2
    m = re.search(r"_frame_(\d+)", base)
    frame_idx = int(m.group(1)) if m else -1
    return prefix, frame_idx

class VarLenVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: napr. dataset/train
        Struktura:
          root_dir/
            class_name/
              PL2_frame_0001.jpg
              PL2_frame_0002.jpg
              PL3_frame_0001.jpg
              ...
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        if len(self.classes) == 0:
            raise RuntimeError(f"No class folders found in: {root_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # samples: list of (list_of_frame_paths, label_idx)
        self.samples = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(IMG_EXTS)]

            videos = defaultdict(list)  # prefix -> list[(frame_idx, path)]
            for f in files:
                prefix, frame_idx = parse_prefix_and_frame(f)
                full_path = os.path.join(cls_path, f)
                videos[prefix].append((frame_idx, full_path))

            for prefix, items in videos.items():
                items_sorted = sorted(items, key=lambda x: x[0])
                frame_paths = [p for _, p in items_sorted]
                if len(frame_paths) == 0:
                    continue
                self.samples.append((frame_paths, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No videos found in: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)  # CxHxW tensor
            else:
                import numpy as np
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        x = torch.stack(frames, dim=0)  # TxCxHxW
        length = x.shape[0]
        y = torch.tensor(label, dtype=torch.long)
        return x, y, length