"""
Drawing canvas interactive component for handwritten digit/word/sentence input.
Uses streamlit-drawable-canvas.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False


def render_drawing_canvas() -> np.ndarray | None:
    """
    Renders an interactive drawing board with stroke controls,
    clear button, and real-time canvas state.
    Returns:
        np.ndarray image or None if empty
    """
    st.markdown("### ✍️ Draw Handwritten Characters or Notes")
    st.caption("Draw digits, characters, words, or full sentences using your mouse, trackpad, or stylus.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        stroke_width = st.slider("Stroke Width", min_value=2, max_value=20, value=6, step=1)
    with col2:
        stroke_color = st.color_picker("Stroke Color", "#000000")
    with col3:
        bg_color = st.color_picker("Background Color", "#FFFFFF")

    if not HAS_CANVAS:
        st.warning("`streamlit-drawable-canvas` is not installed yet. You can upload an image file instead.")
        return None

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=True,
        height=300,
        width=700,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Check if user actually drew anything (non-white/non-transparent pixels)
        img = canvas_result.image_data.astype(np.uint8)
        # Drop alpha channel or check variance
        if np.any(img[:, :, :3] < 250):  # Detected some ink
            return img
    return None
