
from pathlib import Path
from io import BytesIO
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from motion_models.models.mobilenet_lstm import MobileNetV3SmallLSTM
from motion_models.data_utils.transforms import get_val_transforms


# KONFIGURACE

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "motion_models"
    / "reports"
    / "split_subject_v2"
    / "mobilenetv3_lstm"
    / "baseline_10ep"
    / "best.pth"
)

CLASS_NAMES = ["T", "pase"]
NUM_FRAMES = 16
MARGIN = 2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_frames(video_path):
    """
    Výběr 16 snímků podle preprocess_subject_v2.py.
    """
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video neexistuje: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít video: {video_path}")

    frames = []

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
    finally:
        cap.release()

    decoded_frames = len(frames)

    if decoded_frames == 0:
        raise RuntimeError("Nepodařilo se dekódovat video.")

    start_frame = MARGIN
    end_frame = decoded_frames - MARGIN - 1

    if end_frame <= start_frame:
        raise ValueError("Video je příliš krátké.")

    frame_indices = np.linspace(
        start_frame,
        end_frame,
        NUM_FRAMES,
        dtype=int
    )

    if len(set(frame_indices)) != NUM_FRAMES:
        raise ValueError(
            "Video nemá dostatek unikátních snímků."
        )

    return [frames[int(i)] for i in frame_indices]


def preprocess_frames(frames):
    transform = get_val_transforms()
    tensors = []

    for frame_bgr in frames:
        # Původní preprocessing ukládal snímky jako JPEG
        success, jpeg_buffer = cv2.imencode(
            ".jpg", frame_bgr
        )

        if not success:
            raise RuntimeError("Nepodařilo se vytvořit JPEG.")

        with Image.open(
            BytesIO(jpeg_buffer.tobytes())
        ) as image:
            image_rgb = image.convert("RGB")
            tensors.append(transform(image_rgb))

    # (16, 3, 112, 112) -> (1, 16, 3, 112, 112)
    return torch.stack(tensors, dim=0).unsqueeze(0)


def load_model(device):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint nebyl nalezen: {MODEL_PATH}"
        )

    model = MobileNetV3SmallLSTM(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def predict_video(video_path):
    device = get_device()

    print(f"Zařízení: {device}")
    print(f"Vstupní video: {video_path}")

    frames = extract_frames(video_path)
    video_tensor = preprocess_frames(frames)

    print(f"Vstupní tensor: {tuple(video_tensor.shape)}")

    model = load_model(device)
    video_tensor = video_tensor.to(device)

    outputs = model(video_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    probabilities = probabilities.cpu().tolist()

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]

    print("\n--- VÝSLEDEK KLASIFIKACE ---")
    print(f"Predikovaný prvek: {predicted_label}")

    for name, score in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {score * 100:.2f} %")

    return predicted_label, probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rozpoznávání prvků moderní gymnastiky"
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Cesta ke krátkému vstupnímu videu"
    )

    args = parser.parse_args()
    predict_video(args.video)
