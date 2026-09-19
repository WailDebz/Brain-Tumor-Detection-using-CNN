# Developer Guide & Local Setup

This document provides setup and execution instructions for developing, testing, and running the Brain Tumor Detection project.

---

## 1. Prerequisites & Environment Setup

- **Python Version:** Python 3.10 or 3.11 is recommended.
- **Operating System:** Linux, macOS, or Windows.

### Recommended: Creating a Virtual Environment

Using standard Python `venv`:

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

---

## 2. Dependency Installation

Install runtime dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For development and running unit tests:

```bash
pip install -e ".[dev]"
```

---

## 3. Repository Structure & Artifacts

- **Model Artifact Location:**  
  The trained Keras model is expected at `models/brain_tumor_detector.keras`.  
  The application automatically resolves this path relative to the project root.
- **Original Notebook Location:**  
  The historical reference notebook is archived at `notebooks/archive/original_brain_tumor_detection.ipynb`.

---

## 4. Dataset Preparation & Leak-Free Splitting

To prepare, validate, deduplicate, and split a local dataset:

1. Place your raw brain MRI scan images into `data/raw/no/` and `data/raw/yes/`.
2. Run the dataset preparation and validation script:

```bash
python scripts/prepare_data.py --data-dir data/raw --output-dir data/processed
```

This CLI utility will:
- Validate each image file and filter corrupt or empty files.
- Compute SHA-256 hashes to detect and prevent duplicate sample leakage.
- Generate stratified Train (70%), Validation (15%), and Test (15%) splits on original scans.
- Assert zero sample or hash overlap across partitions.
- Output split manifest CSVs into `data/processed/`.

---

## 5. Training the Corrected Baseline CNN Model

To train the modernized baseline CNN on the leak-free dataset partitions:

```bash
python scripts/train.py --data-dir data/raw --epochs 30 --batch-size 32
```

This training pipeline will:
- Partition the dataset using the leak-free stratified split ($70\%$ train, $15\%$ val, $15\%$ test).
- Apply online data augmentation strictly to the training dataset.
- Train the modernized CNN architecture with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.
- Monitor validation loss (the held-out test partition is never accessed during training).
- Save the best model artifact to `models/brain_tumor_cnn_baseline.keras`.
- Export training history to `reports/metrics/training_history_baseline.json` and loss/accuracy curves to `reports/figures/training_curves_baseline.png`.

---

## 6. Running the Streamlit Web Application

Launch the interactive Streamlit interface from the repository root:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 7. Running the Test Suite

Execute the unit test suite with `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```


