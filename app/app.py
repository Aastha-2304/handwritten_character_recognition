"""
Main Streamlit Application for Handwritten Character & Text Recognition.
Features:
- Live Interactive Canvas (draw letters, words, sentences)
- Image Upload (photos, scanned notes, documents)
- End-to-end OCR with Bounding Box overlays
- Multi-engine support (CRNN + CTC, TrOCR)
- Export to TXT, JSON, and PDF
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.components.canvas import render_drawing_canvas
from app.components.export import render_export_buttons
from app.components.visualizer import render_recognition_results
from src.inference.pipeline import OCRPipeline

# Streamlit Page Config
st.set_page_config(
    page_title="Handwritten Character & Text Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_file = ROOT_DIR / "app" / "static" / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.markdown('<div class="hero-title">✍️ Handwritten Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Recognize handwritten characters, single words, and multi-line notes in real-time.</div>', unsafe_allow_html=True)

    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ OCR Engine Configuration")
    engine = st.sidebar.selectbox(
        "Recognition Model",
        options=["trocr", "crnn"],
        index=0,
        help="TrOCR is a Vision-Transformer model. CRNN is a lightweight CNN+BiGRU+CTC network.",
    )

    segment_level = st.sidebar.radio(
        "Segmentation Granularity",
        options=["line", "word"],
        index=0,
        help="'line' extracts entire text lines. 'word' segments individual words inside each line.",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tip**: For drawn characters or short handwritten snippets, choose **line** segmentation."
    )

    # Initialize Pipeline
    pipeline = OCRPipeline(engine=engine)

    # --- Input Mode Selector ---
    tab_canvas, tab_upload = st.tabs(["🖌️ Interactive Canvas", "📁 Upload Image"])

    input_image = None

    with tab_canvas:
        canvas_img = render_drawing_canvas()
        if canvas_img is not None:
            input_image = canvas_img

    with tab_upload:
        st.markdown("### 📤 Upload Handwritten Note or Document")
        uploaded_file = st.file_uploader(
            "Choose an image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            input_image = np.array(pil_img)
            st.image(input_image, caption="Uploaded Image", width=400)

    st.markdown("---")

    # Run OCR button
    if input_image is not None:
        if st.button("🚀 Transcribe Handwriting", type="primary", use_container_width=True):
            with st.spinner("Processing image and running recognition..."):
                result = pipeline.run(input_image, segment_level=segment_level)
                st.session_state["ocr_result"] = result
                st.session_state["input_image"] = input_image

    if "ocr_result" in st.session_state and "input_image" in st.session_state:
        res = st.session_state["ocr_result"]
        img = st.session_state["input_image"]
        render_recognition_results(res, img)
        render_export_buttons(res)


if __name__ == "__main__":
    main()
