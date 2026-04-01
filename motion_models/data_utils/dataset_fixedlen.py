from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

import random
from PIL import ImageEnhance


class FixedLenVideoDataset(Dataset):
    def __init__(self, root_dir, class_names=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if class_names is None:
            self.class_names = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
        else:
            self.class_names = class_names

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for video_dir in sorted(class_dir.iterdir()):

                if not video_dir.is_dir():
                    continue

                frame_paths = sorted(video_dir.glob("frame_*.jpg"))

                if len(frame_paths) != 16:
                    continue

                self.samples.append(
                    (frame_paths, self.class_to_idx[class_name], video_dir.name)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_paths, label, video_id = self.samples[idx]

        #náhodné hodnoty pro celý klip
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)

        frames = []

        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")

            #brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)

            #contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        video_tensor = torch.stack(frames, dim=0)

        return video_tensor, label, video_id