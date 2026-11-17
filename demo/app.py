import json
import sys
import os
import gdown
from pathlib import Path

if not os.path.exists("checkpoints") or len(os.listdir("checkpoints")) == 0:
    with st.spinner("Downloading model checkpoints from Google Drive..."):
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1Hb7WIvnyWMhRCj_QI9KuAfepi_DxeZH2?usp=sharing",
            output="checkpoints",
            quiet=False,
            use_cookies=False
        )

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.autoencoder import CNNAutoencoder
from src.models.efficientad import EfficientADModel
from src.models.patchcore import PatchCoreModel

st.set_page_config(
    page_title="Industrial Visual Anomaly Detection",
    page_icon="🏭",
    layout="wide",
)

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_calibrated_threshold(model_name: str, category: str) -> float:
    res_path = ROOT_DIR / "experiments" / "results" / f"{model_name}_{category}_s42_sz256.json"
    if res_path.exists():
        with open(res_path, "r") as f:
            data = json.load(f)
            if "threshold" in data:
                return float(data["threshold"])
    return 18.5 if model_name == "patchcore" else (0.70 if model_name == "efficientad" else 0.50)


@st.cache_resource
def load_trained_model(model_name: str, category: str):
    ckpt_path = ROOT_DIR / "checkpoints" / model_name / f"{category}_seed42.pt"
    if not ckpt_path.exists():
        return None

    state = torch.load(ckpt_path, map_location=DEVICE)

    if model_name == "patchcore":
        model = PatchCoreModel().to(DEVICE)
        model.memory_bank = state["memory_bank"].to(DEVICE)
        model.load_state_dict(state, strict=False)
    elif model_name == "efficientad":
        model = EfficientADModel().to(DEVICE)
        model.load_state_dict(state)
    else:
        model = CNNAutoencoder().to(DEVICE)
        model.load_state_dict(state)

    model.eval()
    return model


def create_heatmap_overlay(rgb_img: np.ndarray, anomaly_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amap_resized = cv2.resize(anomaly_map, (rgb_img.shape[1], rgb_img.shape[0]))
    amap_norm = (amap_resized - amap_resized.min()) / (amap_resized.max() - amap_resized.min() + 1e-8)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * amap_norm), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_img, 0.55, heatmap_rgb, 0.45, 0)
    return heatmap_rgb, overlay


st.title("🏭 AI-Powered Industrial Visual Anomaly Detection & Defect Localization")

col1, col2 = st.columns([1, 2.2], gap="large")

with col1:
    st.subheader("1. Configuration")
    category = st.selectbox("Select Product Category", ALL_CATEGORIES, index=0)
    model_choice = st.selectbox("Select Model", ["patchcore", "efficientad", "autoencoder"], index=0)

    calibrated_thresh = get_calibrated_threshold(model_choice, category)
    max_slider = max(35.0, round(calibrated_thresh * 2.0, 2))
    threshold = st.slider(
        "Anomaly Decision Threshold (Auto-Calibrated)",
        min_value=0.0,
        max_value=max_slider,
        value=round(calibrated_thresh, 4),
        step=0.01,
    )

    uploaded_file = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("2. Inspection Results")
    if uploaded_file is None:
        st.info("Upload a product image from the left panel to run defect inspection.")
    else:
        model = load_trained_model(model_choice, category)
        if model is None:
            st.error(f"Checkpoint not found: checkpoints/{model_choice}/{category}_seed42.pt")
        else:
            pil_img = Image.open(uploaded_file).convert("RGB")
            rgb_np = np.array(pil_img.resize((256, 256)))
            input_tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                scores, anomaly_maps = model.predict_anomaly(input_tensor)
                score = float(scores.squeeze().cpu().item())
                amap = anomaly_maps.squeeze().cpu().numpy()

            is_anomalous = score >= threshold
            st.caption("Inspection Status")
            if is_anomalous:
                st.error(f"🚨 ANOMALOUS (Defect Detected) — Score: {score:.4f} | Threshold: {threshold:.4f}")
            else:
                st.success(f"✅ NORMAL (Defect-Free) — Score: {score:.4f} | Threshold: {threshold:.4f}")

            heatmap_rgb, overlay_rgb = create_heatmap_overlay(rgb_np, amap)

            img_col1, img_col2, img_col3 = st.columns(3)
            with img_col1:
                st.image(rgb_np, caption="Original Image", use_container_width=True)
            with img_col2:
                st.image(heatmap_rgb, caption="Predicted Anomaly Map", use_container_width=True)
            with img_col3:
                st.image(overlay_rgb, caption="Defect Localization Overlay", use_container_width=True)