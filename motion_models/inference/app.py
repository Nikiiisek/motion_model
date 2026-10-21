
from pathlib import Path
import tempfile
import time

import streamlit as st
import torch

from motion_models.inference.predict_video import (
    CLASS_NAMES,
    extract_frames,
    preprocess_frames,
    load_model,
    get_device,
)


st.set_page_config(
    page_title="Prediction of Gymnastics",
    layout="centered",
)

st.title("Prediction of Gymnastics")
st.write("Klasifikace gymnastických prvků z videozáznamu")
st.divider()


# Načtený model zůstává v paměti mezi jednotlivými analýzami.
@st.cache_resource
def get_cached_model():
    device = get_device()
    model = load_model(device)
    return model, device


def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


st.subheader("Vstupní video")

uploaded_file = st.file_uploader(
    "Vyber video",
    type=["mov", "mp4", "avi", "mkv"],
)

if uploaded_file is not None:
    st.video(uploaded_file)

    st.caption(
        "Aktuální model klasifikuje dva prvky: T a pasé."
    )

    if st.button("Spustit klasifikaci", type="primary"):

        temporary_path = None

        try:
            with st.spinner("Probíhá zpracování videa..."):

                suffix = Path(uploaded_file.name).suffix.lower()

                # OpenCV načítá video z dočasného souboru.
                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temporary_path = Path(tmp.name)

                start_total = time.perf_counter()

                start = time.perf_counter()
                frames = extract_frames(temporary_path)
                decode_time = time.perf_counter() - start

                start = time.perf_counter()
                video_tensor = preprocess_frames(frames)
                preprocessing_time = time.perf_counter() - start

                start = time.perf_counter()
                model, device = get_cached_model()
                model_load_time = time.perf_counter() - start

                video_tensor = video_tensor.to(device)
                synchronize_device(device)

                start = time.perf_counter()

                with torch.inference_mode():
                    outputs = model(video_tensor)
                    probabilities = torch.softmax(
                        outputs, dim=1
                    )[0]

                synchronize_device(device)
                inference_time = time.perf_counter() - start

                scores = probabilities.cpu().tolist()

                predicted_index = max(
                    range(len(scores)),
                    key=lambda i: scores[i],
                )
                predicted_label = CLASS_NAMES[predicted_index]

                total_time = time.perf_counter() - start_total

            st.divider()
            st.subheader("Výsledek klasifikace")

            st.write(f"**Predikovaná třída:** {predicted_label}")

            for class_name, score in zip(CLASS_NAMES, scores):
                st.write(f"{class_name}: {score * 100:.2f} %")
                st.progress(float(score))

            st.caption(
                "Skóre modelu nepředstavuje garantovanou "
                "pravděpodobnost správné klasifikace."
            )

            st.divider()
            st.subheader("Informace o zpracování")

            st.write(f"Použitý model: MobileNetV3 Small + LSTM")
            st.write(f"Výpočetní zařízení: {device}")
            st.write(f"Celková doba zpracování: {total_time:.3f} s")
            st.write(f"Doba inference: {inference_time:.3f} s")

            with st.expander("Podrobnosti"):
                st.write(f"Dekódování videa: {decode_time:.3f} s")
                st.write(
                    f"Předzpracování: {preprocessing_time:.3f} s"
                )
                st.write(f"Načtení modelu: {model_load_time:.3f} s")
                st.write(f"Počet snímků: {len(frames)}")
                st.write(
                    f"Rozměry tensoru: {tuple(video_tensor.shape)}"
                )

        except Exception as error:
            st.error("Video se nepodařilo zpracovat.")
            st.exception(error)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
