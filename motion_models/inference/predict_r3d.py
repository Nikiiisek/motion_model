
import argparse
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from motion_models.models.R3D18 import R3D18Classifier
from motion_models.inference.predict_video import (
    extract_frames,
    get_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "motion_models/reports/split_subject_v2/"
    "r3d18/baseline_10ep/best.pth"
)

CLASS_NAMES = ["T", "pase"]

# Stejné předzpracování jako při vyhodnocení R3D-18.
r3d_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645],
        std=[0.22803, 0.22145, 0.216989],
    ),
])


def preprocess_r3d(frames):
    processed = []

    for frame in frames:
        success, encoded = cv2.imencode(".jpg", frame)

        if not success:
            raise ValueError("Nepodařilo se zpracovat snímek videa.")

        image = Image.open(BytesIO(encoded.tobytes())).convert("RGB")
        processed.append(r3d_transform(image))

    return torch.stack(processed).unsqueeze(0)


def load_r3d(device):
    model = R3D18Classifier(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def predict_r3d(video_path):
    device = get_device()

    print(f"Zařízení: {device}")
    print(f"Vstupní video: {video_path}")

    frames = extract_frames(Path(video_path))
    video_tensor = preprocess_r3d(frames).to(device)

    print(f"Vstupní tensor: {tuple(video_tensor.shape)}")

    model = load_r3d(device)
    outputs = model(video_tensor)

    probabilities = torch.softmax(outputs, dim=1)[0]
    scores = probabilities.cpu().tolist()
    predicted_index = max(range(len(scores)), key=lambda i: scores[i])

    print("\n--- VÝSLEDEK R3D-18 ---")
    print(f"Predikovaný prvek: {CLASS_NAMES[predicted_index]}")

    for class_name, score in zip(CLASS_NAMES, scores):
        print(f"{class_name}: {score * 100:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Cesta k videozáznamu")
    args = parser.parse_args()

    predict_r3d(args.video)