# Brain Tumor Classification with CNN

An engineering-focused computer vision project for binary classification of brain axial MRI scans (Tumor vs. No Tumor) using Deep Learning and Convolutional Neural Networks (CNNs).

---

## Overview

This repository provides a modular, reproducible pipeline for brain tumor detection from T1/T2/contrast axial magnetic resonance imaging (MRI) slices, accompanied by an interactive Streamlit inference web application.

---

## Current Status

- **Stage 1 (Current):** Engineering foundation, clean directory layout, dependency management, unit testing, and robust application entrypoints.
- **Stage 2 (Planned):** Leak-free dataset splitting, online data augmentation, model retraining, and benchmark evaluation.
- **Stage 3 (Planned):** Visual explainability (Grad-CAM overlays) and containerization (Docker).

---

## Project Structure

```text
Brain-Tumor-Detection-using-CNN/
├── app/
│   └── streamlit_app.py         # Streamlit web application entrypoint
├── docs/
│   └── assets/                  # UI screenshots and documentation assets
├── models/
│   └── brain_tumor_detector.keras  # Trained baseline Keras model artifact
├── notebooks/
│   └── archive/
│       └── original_brain_tumor_detection.ipynb  # Preserved original reference notebook
├── src/
│   ├── config.py                # Project paths and configuration constants
│   ├── data/                    # Dataset loaders and splitting logic (in development)
│   ├── models/                  # CNN architecture definitions (in development)
│   ├── evaluation/              # Evaluation metrics and reporting (in development)
│   └── inference/
│       ├── preprocessing.py     # Image normalization and resizing pipeline
│       └── predictor.py         # Model loading and inference utilities
├── tests/
│   ├── test_config.py           # Configuration and path resolution tests
│   ├── test_preprocessing.py    # Preprocessing contract unit tests
│   └── test_inference.py        # Inference interface unit tests
├── DEVELOPMENT.md               # Local development and setup guide
├── pyproject.toml               # Build system and tool configuration
├── requirements.txt             # Pinned runtime dependencies
├── .gitignore                   # Standard Python/ML ignore rules
└── README.md                    # Project documentation
```

---

## Installation

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/wail-d/Brain-Tumor-Detection-using-CNN.git
cd Brain-Tumor-Detection-using-CNN

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web interface locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Methodology

The baseline pipeline employs a 2D Convolutional Neural Network:
- **Input:** Axial MRI slices converted to single-channel grayscale and normalized to $90 \times 90$ pixels.
- **Architecture:** 3 Convolutional blocks with ReLU activations, max-pooling, L2 kernel regularization, followed by a Dense layer with Dropout ($0.5$), and a Sigmoid classification unit.
- **Loss & Optimizer:** Binary Cross-Entropy optimized via Adam.

---

## Evaluation

> **Methodological Audit Note:**  
> The originally reported 99% accuracy in the archived notebook is **not** considered a reliable generalization benchmark. The original prototype applied offline data augmentation to the dataset *prior* to performing the train/test split, resulting in identical base patient scans existing across both training and evaluation sets (data leakage).
>
> In the upcoming implementation stage, the model will be retrained on a strictly leak-free, stratified train/validation/test split with online augmentation, and evaluated on comprehensive medical metrics (Sensitivity, Specificity, F1-Score, and ROC-AUC). Revised benchmarks will be reported here once retraining is completed.

---

## Limitations & Disclaimer

- **Educational & Research Prototype:** This software is an engineering demonstration and is **not** a certified medical device or clinical diagnostic tool.
- **Non-Diagnostic:** It must not be used for primary diagnosis, patient screening, or clinical decision support.
- **Data Constraints:** Binary classification over axial MRI slices does not account for full 3D volumetric context or multi-modal imaging series (e.g., FLAIR, T1CE).

