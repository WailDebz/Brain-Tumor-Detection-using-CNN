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

- **Model Artifact Locations:**  
  - Primary Application Model: `models/brain_tumor_mobilenetv2_frozen.keras` (MobileNetV2 feature extractor).
  - Baseline Custom CNN: `models/brain_tumor_cnn_baseline.keras`.
  - Historical Model: `models/brain_tumor_detector.keras` (preserved reference).
- **Original Notebook Location:**  
  The historical reference notebook is archived at `notebooks/archive/original_brain_tumor_detection.ipynb`.

---

## 4. Dataset Preparation & Leak-Free Splitting

To prepare, validate, deduplicate, and partition the dataset using the repository's deterministic pipeline:

1. **Obtain the dataset:** Acquire the Brain MRI image dataset (e.g., from the public Kaggle repository *Brain MRI Images for Brain Tumor Detection*).
2. **Place original files:** Place raw image files under `data/raw/no/` (No Tumor, label 0) and `data/raw/yes/` (Tumor, label 1). Raw images are gitignored and must never be committed.
3. **Execute the preparation CLI:**
   ```bash
   python scripts/prepare_data.py --data-dir data/raw --output-dir data/processed
   ```
4. **Automated processing performed by CLI:**
   - Case-insensitively scans and validates all image files (`.jpg`, `.jpeg`, `.png`), rejecting unreadable or corrupt files.
   - Computes SHA-256 hashes to detect and filter exact byte-identical duplicate files within the raw dataset.
   - Performs stratified Train ($70\%$), Validation ($15\%$), and Test ($15\%$) partitioning on unique original images using a fixed random seed (`seed=42`).
   - Programmatically asserts zero sample path and zero hash overlap across all three partitions.
   - Generates portable, repository-relative manifest CSVs saved to `data/processed/`:
     - `data/processed/train_manifest.csv`
     - `data/processed/val_manifest.csv`
     - `data/processed/test_manifest.csv`
     - `data/processed/full_split_manifest.csv`
5. **No offline augmentation:** The preparation process does **not** create or save augmented images to disk. Online data augmentation is applied exclusively in-memory to training mini-batches during model training.

---

## 5. Model Training

To train the custom baseline CNN on the leak-free dataset partitions:

```bash
python scripts/train.py --model-type cnn --data-dir data/raw --epochs 30 --batch-size 32
```

To train the MobileNetV2 frozen feature extractor:

```bash
python scripts/train.py --model-type transfer_mobilenetv2 --data-dir data/raw --epochs 30 --batch-size 32
```

To train the controlled fine-tuning MobileNetV2 model:

```bash
python scripts/train.py --model-type transfer_mobilenetv2_finetuned --data-dir data/raw --epochs 30 --lr 1e-5
```

The training pipeline will:
- Partition the dataset using the leak-free stratified split ($70\%$ train, $15\%$ val, $15\%$ test).
- Apply online data augmentation strictly to the training dataset.
- Train with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.
- Monitor validation loss (the held-out test partition is never accessed during training).
- Save the best model artifact to `models/`.
- Export training history to `reports/metrics/` and loss/accuracy curves to `reports/figures/`.

---

## 6. Held-Out Test Set Evaluation

To evaluate a trained model on the held-out test split with the default fixed threshold ($0.5$):

```bash
python scripts/evaluate.py \
    --model-path models/brain_tumor_cnn_baseline.keras \
    --test-manifest data/processed/test_manifest.csv
```

To evaluate using validation-based threshold optimization (optimizing Youden's J statistic on the validation partition, then freezing it for test evaluation):

```bash
python scripts/evaluate.py \
    --model-path models/brain_tumor_cnn_baseline.keras \
    --test-manifest data/processed/test_manifest.csv \
    --val-manifest data/processed/val_manifest.csv \
    --tune-threshold \
    --tuning-metric youden_j
```

This evaluation pipeline will:
- Run deterministic, unaugmented inference on the pristine test partition.
- Calculate clinical metrics: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, ROC-AUC, and PR-AUC.
- Export machine-readable metrics to `reports/metrics/test_metrics_{model_name}_baseline.json` and `test_metrics_{model_name}_baseline.csv`.
- Generate model-specific diagnostic plots in `reports/figures/`:
  - Confusion Matrix (`reports/figures/confusion_matrix_test_{model_name}.png`)
  - ROC Curve (`reports/figures/roc_curve_test_{model_name}.png`)
  - Precision-Recall Curve (`reports/figures/pr_curve_test_{model_name}.png`)

---

## 7. Running the Streamlit Web Application

Launch the interactive Streamlit application from the repository root:

```bash
# Using the virtual environment's Python interpreter:
python -m streamlit run app/streamlit_app.py

# Or on Windows PowerShell:
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The web interface will open in your default browser at `http://localhost:8501`.

### Application Summary:
- **Primary Model:** MobileNetV2 frozen feature extractor (`models/brain_tumor_mobilenetv2_frozen.keras`), cached with `@st.cache_resource`.
- **Input Validation:** Checks uploaded files for valid image headers, non-zero dimensions, and formats (RGB, RGBA, Grayscale).
- **Inference Pipeline:** Reuses `src/data/preprocessing.py` for standard $(90, 90, 1)$ normalization in $[0, 1]$.
- **Prediction Display:** Reports predicted class (`Tumor` or `No Tumor`) and continuous model predicted probability at the fixed $0.50$ decision threshold.
- **Explainability:** Displays 3-panel Grad-CAM saliency visualizations (Input Scan, Heatmap, Overlay) computed via `src/explainability/gradcam.py`.
- **Safety:** Displays a non-diagnostic research disclaimer.

---

## 8. Running the Test Suite

Execute the unit test suite with `pytest`:

```bash
# Fast quiet run:
python -m pytest -q

# Or with verbose test breakdown:
python -m pytest -v
```


