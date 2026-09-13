"""
Visualization components:
Draws detected line & word bounding boxes with labels and confidence chips.
"""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.inference.pipeline import PipelineResult


def draw_bounding_boxes(
    original_image: np.ndarray,
    result: PipelineResult,
    box_color: tuple[int, int, int] = (108, 92, 231), # #6C5CE7 in RGB
    thickness: int = 2,
) -> np.ndarray:
    """
    Renders bounding boxes over the input image.
    """
    vis = original_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    elif vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGBA2RGB)

    for line in result.lines:
        # Draw line bounding box
        lb = line.line_box
        cv2.rectangle(vis, (lb.x, lb.y), (lb.x + lb.w, lb.y + lb.h), (255, 107, 129), 1)

        # Draw word bounding boxes
        for word in line.words:
            wb = word.box
            cv2.rectangle(vis, (wb.x, wb.y), (wb.x + wb.w, wb.y + wb.h), box_color, thickness)

    return vis


def render_recognition_results(result: PipelineResult, original_image: np.ndarray):
    """
    Displays the visual overlay, segmentations, and recognized text blocks.
    """
    st.markdown("### 🔍 Segmentation & Detection")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original with Detected Regions**")
        overlay = draw_bounding_boxes(original_image, result)
        st.image(overlay, use_column_width=True)

    with col2:
        st.markdown("**Preprocessed & Cleaned Binary**")
        # Invert for human-friendly viewing (black handwriting on white paper)
        display_binary = 255 - result.cleaned_binary if result.cleaned_binary.ndim == 2 else result.cleaned_binary
        st.image(display_binary, caption="Binarized (Black ink on white paper)", use_column_width=True)

    st.markdown("---")
    st.markdown("### 📝 Recognized Text")

    if not result.full_text.strip():
        st.info("No text detected or recognition returned empty.")
    else:
        st.text_area(
            "Transcribed Content",
            value=result.full_text,
            height=150,
            help="You can review and copy the transcript.",
        )

        st.markdown("**Line-by-Line Breakdown:**")
        for idx, line in enumerate(result.lines):
            with st.expander(f"Line {idx + 1}: '{line.text}'", expanded=True):
                st.write(f"**Text:** `{line.text}`")
                st.write(f"**Bounding Box:** x={line.line_box.x}, y={line.line_box.y}, w={line.line_box.w}, h={line.line_box.h}")
