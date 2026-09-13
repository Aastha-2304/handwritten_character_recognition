# Handwritten Character & Text Recognition System

An end-to-end Handwritten Text Recognition (HTR) and Optical Character Recognition (OCR) platform supporting character, word, and multi-line paragraph recognition.

---

## 🌟 Key Features

- **Interactive Canvas (`Streamlit`)**: Draw digits, individual handwritten characters, or complete sentences with customizable brush width and colors.
- **Image Upload & Scanner**: Upload scanned notes, camera snapshots, or forms.
- **Robust Preprocessing**:
  - Denoising (Fast Non-Local Means)
  - Adaptive Gaussian thresholding (Otsu & local adaptive)
  - Deskewing via minimum bounding area rotation correction
- **Intelligent Segmentation**:
  - **Horizontal Projection Profile** for text line segmentation
  - **Vertical Projection Profile** with adaptive writer spacing estimation for word segmentation
- **Dual Neural Network Engines**:
  - **CRNN (CNN + Bidirectional GRU + CTC)**: Ultra-fast, lightweight inference architecture designed for edge devices and fast CPU/GPU transcription.
  - **TrOCR (Vision-Encoder-Decoder Transformer)**: High-accuracy handwritten transcription powered by pretrained Vision Transformers.
- **Multi-Format Export**: One-click download of transcriptions as **Plain Text (.txt)**, **JSON (.json)** with coordinates and bounding boxes, and **PDF Reports (.pdf)**.

---

## 📁 Repository Structure

```
handwritten-ocr/
├── .streamlit/
│   └── config.toml               # Streamlit styling & max upload config
├── app/
│   ├── app.py                    # Main interactive Streamlit application
│   ├── components/
│   │   ├── canvas.py             # Drawing canvas component
│   │   ├── visualizer.py         # Bounding box & segmentation visualizer
│   │   └── export.py             # PDF, JSON, TXT download generators
│   └── static/
│       └── style.css             # UI styling & glassmorphism theme
├── src/
│   ├── preprocessing/
│   │   ├── image_ops.py          # Grayscale, denoise, binarize, deskew
│   │   ├── segmentation.py       # Projection-profile line & word segmentation
│   │   └── transforms.py         # Aspect-ratio preserving padding & tensors
│   ├── models/
│   │   ├── crnn.py               # CRNN PyTorch neural network
│   │   ├── ctc_decoder.py        # CTC greedy best-path decoder
│   │   ├── trocr_wrapper.py      # HuggingFace TrOCR integration
│   │   └── train_crnn.py         # PyTorch training loop for CRNN
│   ├── inference/
│   │   ├── predictor.py          # Multi-engine inference controller
│   │   └── pipeline.py           # End-to-end pipeline (Image -> Text)
│   └── utils/
│       ├── config.py             # Central paths, charset, and constants
│       └── metrics.py            # CER, WER, and Levenshtein distance
└── tests/
    ├── test_preporcessing_somking.py # Segmentation smoke test
    ├── test_ctc_decoder.py           # CTC decoding test
    └── test_crnn_architecture.py     # Neural network forward pass test
```

---

## 🚀 Getting Started

### 1. Activate Environment & Install Dependencies

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Tests

```powershell
python tests/test_preporcessing_somking.py
python tests/test_ctc_decoder.py
python tests/test_crnn_architecture.py
```

### 3. Launch Web Application

```powershell
streamlit run app/app.py
```
