# Brain Tumor Classification with CNN

This project classifies 2D axial brain MRI images into two classes: **No Tumor** (label 0) and **Tumor** (label 1).

It started from an earlier CNN notebook project. I rebuilt the data pipeline to remove data leakage, trained a modernized baseline CNN, added a transfer learning model using MobileNetV2, and added Grad-CAM visual explanations and a Streamlit interface.

---

## What the Project Does

The pipeline takes an axial brain MRI image, converts it to 90×90 grayscale, normalizes pixel values to [0, 1], and predicts the probability of tumor presence.

The current primary model used in the application is an ImageNet-pretrained **MobileNetV2** configured as a frozen feature extractor with a trained classification head (`models/brain_tumor_mobilenetv2_frozen.keras`). The model handles in-graph replication from 1 grayscale channel to 3 channels and scales inputs to the [-1, 1] range expected by MobileNetV2.

The earlier custom CNN (`models/brain_tumor_cnn_baseline.keras`) and the original notebook and weights (`models/brain_tumor_detector.keras`) are preserved in the repository for reference.

---

## Dataset and Data Preparation

The raw dataset comes from the public Kaggle *Brain MRI Images for Brain Tumor Detection* collection, containing 253 image files:
- `no/` (No Tumor): 98 images
- `yes/` (Tumor): 155 images

### Exact Duplicate Removal
Scanning the files with SHA-256 hashing identified 25 exact byte-for-byte duplicate images in the source data (11 in `no/` and 14 in `yes/`). Removing these duplicates left **228 unique image files**:
- No Tumor: 87 images (38.2%)
- Tumor: 141 images (61.8%)

### Split
The 228 unique images are split into train, validation, and test sets using stratified sampling and a fixed random seed of 42:
- **Train:** 159 images (61 No Tumor, 98 Tumor) — 69.7%
- **Validation:** 34 images (13 No Tumor, 21 Tumor) — 14.9%
- **Test:** 35 images (13 No Tumor, 22 Tumor) — 15.4%

Data augmentation (horizontal flips, small rotations, translations, and zooms) is applied **online during training only**. Validation and test images are never augmented.

**Metadata Note:** Patient-level independence could not be established from the available source metadata. The source dataset contains individual 2D image slices without patient IDs or DICOM headers.

---

## Data Leakage in the Original Prototype

The original student notebook reported roughly 99% test accuracy. However, that notebook applied offline data augmentation to the full dataset *before* the train/test split. Because copies of the same original images were generated on disk first, augmented variants of the same image ended up in both the training set and the test set. Duplicate images in the source data were also not removed.

For this reason, the original 99% accuracy is not a valid generalization measurement and is not used as the benchmark for this project. The original notebook is kept in `notebooks/archive/original_brain_tumor_detection.ipynb` and its exported model in `models/brain_tumor_detector.keras` as historical reference files.

---

## Current Model

The primary model used by the application is `models/brain_tumor_mobilenetv2_frozen.keras`:

- **Backbone:** MobileNetV2 pretrained on ImageNet (frozen, 2,257,984 non-trainable parameters)
- **Input:** 90×90 grayscale, adapted in-graph to 3 channels and scaled to [-1, 1]
- **Classification Head:**
  - `GlobalAveragePooling2D()` (1280 features)
  - `Dense(64, ReLU, L2=1e-4)` (81,984 parameters)
  - `Dropout(0.3)`
  - `Dense(1, Sigmoid)` (65 parameters)
- **Total Parameters:** 2,340,033
- **Trainable Parameters:** 82,049

A fine-tuned MobileNetV2 model (with top backbone layers unfrozen at a learning rate of 1e-5) was also trained as an experiment, but the application defaults to the frozen feature extractor because it achieved equivalent validation performance with fewer trainable parameters.

---

## Training

The models were trained with the following setup:
- Random seed: 42
- Optimizer: Adam (learning rate = 0.001 for frozen feature extraction)
- Loss: Binary Cross-Entropy
- Batch size: 32
- Maximum epochs: 30
- Callbacks: `EarlyStopping` (patience = 8, monitoring validation loss), `ReduceLROnPlateau` (patience = 4, factor = 0.5), `ModelCheckpoint` (saving best validation loss weights)
- The test set was not passed to `model.fit()` or used for early stopping.

---

## Evaluation

The models were evaluated across several classification metrics:

| Metric | MobileNetV2 Frozen (Threshold = 0.50) | Custom CNN Baseline (Threshold = 0.50) |
| :--- | :---: | :---: |
| **Accuracy** | **85.71%** | 62.86% |
| **Sensitivity (Recall)** | **86.36%** (19/22) | 100.00% (22/22) |
| **Specificity** | **84.62%** (11/13) | 0.00% (0/13) |
| **Precision** | **90.48%** (19/21) | 62.86% (22/35) |
| **F1-Score** | **88.37%** | 77.19% |
| **ROC-AUC** | **0.9231** | 0.7902 |
| **PR-AUC** | **0.9657** | 0.8089 |

Confusion matrix counts on the 35-image test split for MobileNetV2 at threshold 0.50:
- True Positives (TP): 19
- True Negatives (TN): 11
- False Positives (FP): 2
- False Negatives (FN): 3

**Methodological Note:** The 35-image test split was used during model development to compare the custom CNN and MobileNetV2 feature-extraction experiments. Because architecture comparison was informed by this test partition, these figures should be viewed as an exploratory benchmark rather than an untouched final measurement of real-world generalization.

---

## Grad-CAM Visual Explanations

The project includes Gradient-weighted Class Activation Mapping (Grad-CAM) to provide qualitative visual explanations for individual predictions.

- **Target Layer:** MobileNetV2 `out_relu` (the rectified output of the final convolutional layer `Conv_1`, feature map size 3×3×1280).
- **Target Score:** For Tumor predictions, Grad-CAM computes gradients for the positive tumor score. For No Tumor predictions, it computes gradients for the negative-class score ($1 - p$).
- **Heatmap Display:** The 2D activation map is normalized to [0, 1], resized to the original image dimensions, and overlaid using OpenCV's Jet colormap with a blending factor of $\alpha = 0.45$.

**Interpretation Note:** Grad-CAM heatmaps show spatial regions that contributed positively to the model's output score for the chosen class. They do not represent medical segmentations or exact tumor margins.

---

## Streamlit Application

The interactive web interface is in `app/streamlit_app.py`:

- Accepts JPG, JPEG, and PNG uploads in RGB, RGBA, and grayscale formats.
- Preprocesses images using the shared `src/data/preprocessing.py` functions.
- Runs inference through the cached MobileNetV2 model at the fixed 0.50 threshold.
- Displays the predicted class (`Tumor` or `No Tumor`) and the **Model predicted probability**.
- Generates and displays a 3-panel Grad-CAM visualization (Input Scan, Heatmap, Overlay).
- Includes an "About the Model" expandable section and a non-diagnostic research disclaimer.

Launch the application:

```bash
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

---

## Project Structure

```text
Brain-Tumor-Detection-using-CNN/
├── app/
│   └── streamlit_app.py               # Streamlit web application
├── data/
│   ├── raw/                           # Raw image files (no/ and yes/, gitignored)
│   └── processed/                     # Manifest CSVs for train, val, and test splits
├── docs/
│   ├── assets/                        # Screenshots and UI assets
│   ├── experiments/                   # Experiment notes for transfer learning and Grad-CAM
│   └── leakage_analysis.md            # Documented audit of the original pipeline leakage
├── models/
│   ├── brain_tumor_detector.keras     # Original model (historical reference)
│   ├── brain_tumor_cnn_baseline.keras # Custom CNN baseline model
│   ├── brain_tumor_mobilenetv2_frozen.keras   # Primary application model
│   └── brain_tumor_mobilenetv2_finetuned.keras# Fine-tuned experiment model
├── notebooks/
│   └── archive/
│       └── original_brain_tumor_detection.ipynb # Original notebook (historical reference)
├── reports/
│   ├── figures/                       # Training curves, ROC, PR, confusion matrices, Grad-CAM
│   └── metrics/                       # JSON and CSV logs for training and evaluation
├── scripts/
│   ├── compile_comparison.py          # Builds model comparison JSON and CSV tables
│   ├── evaluate.py                    # CLI for model evaluation on test manifests
│   ├── generate_gradcam_example.py    # CLI to generate Grad-CAM figure panels
│   ├── generate_model_figures.py      # Generates model-specific evaluation plots
│   ├── prepare_data.py                # Ingestion, deduplication, and split CLI
│   ├── smoke_test_app.py              # Component smoke test for Streamlit workflows
│   ├── train.py                       # CLI for training CNN and transfer models
│   └── verify_env.py                  # Environment verification utility
├── src/
│   ├── config.py                      # Project paths, constants, and hyperparameters
│   ├── data/                          # Ingestion, deduplication, splitting, and loader
│   ├── evaluation/                    # Metrics, comparison tables, and visualizers
│   ├── explainability/                # Grad-CAM computation and overlay logic
│   ├── inference/                     # Predictor and preprocessing interfaces
│   ├── models/                        # Custom CNN and MobileNetV2 model builders
│   └── utils/                         # Reproducibility seed utilities
├── tests/                             # Pytest test suite (86 tests)
├── DEVELOPMENT.md                     # Developer setup and execution guide
├── pyproject.toml                     # Build configuration and project metadata
├── requirements.txt                   # Pinned runtime dependencies
├── .gitignore                         # Python, data, and cache ignore rules
└── README.md                          # Project documentation
```

---

## Installation / Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/wail-d/Brain-Tumor-Detection-using-CNN.git
   cd Brain-Tumor-Detection-using-CNN
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Start the Streamlit interface locally:

```bash
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Running Tests

Run the full pytest suite:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite contains 86 tests covering:
- Image preprocessing (RGB, RGBA, grayscale, bounds, invalid inputs)
- Data ingestion, SHA-256 duplicate detection, and manifest creation
- Stratified splitting reproducibility and zero-leakage assertions
- Custom CNN and MobileNetV2 model architectures and parameter counts
- Training loop callbacks and history logging
- Evaluation metrics (Sensitivity, Specificity, F1, ROC-AUC, PR-AUC)
- Grad-CAM gradient tape computation and overlay blending
- Streamlit input validation and prediction formatting

---

## Limitations

- **Small Dataset:** The dataset contains 228 unique images across all splits. With 35 test samples, each classification error changes accuracy by roughly 2.86%.
- **2D Slices:** The models operate on individual 2D axial slices rather than full 3D volumetric MRI scans.
- **Patient-Level Independence:** The source Kaggle dataset lacks patient identifiers and DICOM headers, so deduplication was performed at the file level via SHA-256 hashing. Multiple slices may come from the same patient scan volume.
- **Exploratory Benchmark:** The 35-image test split was used to compare the custom CNN and MobileNetV2 models during development.
- **Grad-CAM Resolution:** Grad-CAM heatmaps are derived from a 3×3 feature map and upsampled, providing coarse qualitative saliency rather than precise tissue boundaries.
- **Experimental Models:** Model predictions are experimental output scores and are not calibrated clinical probabilities.

---

## Disclaimer

Educational / research prototype. This application is not a medical device and must not be used for diagnosis, patient screening, or clinical decision-making. Model outputs are experimental and may be incorrect.


