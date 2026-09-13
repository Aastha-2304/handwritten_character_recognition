"""
Export utilities:
Allows exporting recognized transcription to Plain Text (.txt),
JSON (.json) with bounding box metadata, and styled PDF (.pdf).
"""
from __future__ import annotations

import io
import json
import streamlit as st
from src.inference.pipeline import PipelineResult

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


def export_as_txt(text: str) -> bytes:
    return text.encode("utf-8")


def export_as_json(result: PipelineResult) -> bytes:
    data = {
        "full_text": result.full_text,
        "lines": [
            {
                "line_text": line.text,
                "bbox": {"x": line.line_box.x, "y": line.line_box.y, "w": line.line_box.w, "h": line.line_box.h},
                "words": [
                    {
                        "word_text": w.text,
                        "confidence": w.confidence,
                        "bbox": {"x": w.box.x, "y": w.box.y, "w": w.box.w, "h": w.box.h}
                    }
                    for w in line.words
                ]
            }
            for line in result.lines
        ]
    }
    return json.dumps(data, indent=2).encode("utf-8")


def export_as_pdf(result: PipelineResult, title: str = "Handwritten OCR Transcription Report") -> bytes:
    if not HAS_FPDF:
        return b""

    try:
        from fpdf.enums import XPos, YPos
        has_enums = True
    except ImportError:
        has_enums = False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)

    if has_enums:
        pdf.cell(0, 10, text=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(10)
        pdf.set_font("Helvetica", size=11)
        for i, line in enumerate(result.lines):
            line_str = f"Line {i + 1}: {line.text}"
            safe_str = line_str.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 8, text=safe_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.cell(0, 10, txt=title, ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Helvetica", size=11)
        for i, line in enumerate(result.lines):
            line_str = f"Line {i + 1}: {line.text}"
            safe_str = line_str.encode("latin-1", "replace").decode("latin-1")
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 8, txt=safe_str)
            pdf.set_x(pdf.l_margin)

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


def render_export_buttons(result: PipelineResult):
    """Render export download buttons for TXT, JSON, and PDF."""
    st.markdown("### 📥 Export Transcription")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📄 Download Text (.txt)",
            data=export_as_txt(result.full_text),
            file_name="transcription.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="🗂️ Download JSON (.json)",
            data=export_as_json(result),
            file_name="transcription.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        if HAS_FPDF:
            pdf_bytes = export_as_pdf(result)
            st.download_button(
                label="📑 Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name="transcription_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button("📑 PDF Export (requires fpdf2)", disabled=True, use_container_width=True)
