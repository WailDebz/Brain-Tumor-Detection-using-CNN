# Methodological Audit: Data Leakage Analysis & Pipeline Redesign

This document provides a technical post-mortem of the data leakage flaw present in the original implementation (`notebooks/archive/original_brain_tumor_detection.ipynb`) and explains the architecture of the corrected, leak-free data pipeline.

---

## 1. The Original Pipeline & Root Cause of Artificial 99% Accuracy

### 1.1 Original Workflow Sequence

In the original prototype notebook, the pipeline executed in the following order:

```text
[ 253 Original Brain MRI Scans ]
 (98 "no", 155 "yes")
         │
         ▼
[ Step 1: Offline Disk Augmentation on ALL 253 Images ]
 - 155 positive images × 60 augmented copies = 9,300 files
 - 98 negative images × 120 augmented copies = 11,760 files
 Total dataset artificially inflated: 253 → 21,253 images
         │
         ▼
[ Step 2: Random Train/Test Split (80% / 20%) ]
 Total: 21,253 images → Train: 17,002 images | Test: 4,251 images
         │
         ▼
[ Step 3: Model Training (10 Epochs) ]
         │
         ▼
[ Step 4: Model Evaluation on Test Split ]
 Reported Result: Test Accuracy = 99.03%, Test Loss = 0.1968
```

### 1.2 Mathematical & Diagnostic Failure Mode

When random sampling ($80/20$) is performed on a pool containing 60 to 120 synthetic variants of each base image:

- For any given base patient image $I_k$, let its augmented variants be $V(I_k) = \{v_{k, 1}, v_{k, 2}, \dots, v_{k, M}\}$ where $M \in \{60, 120\}$.
- The probability that **all** variants of $I_k$ fall strictly into the training split and none into the test split is:
  $$P(\text{No leakage for } I_k) = (0.80)^M \approx (0.80)^{60} \approx 1.53 \times 10^{-6}$$
- Consequently, with $253$ base scans, **100% of test samples had identical or near-identical augmented twins in the training set**.
- The resulting $99.03\%$ test accuracy did **not** measure clinical generalization or tumor feature abstraction; it merely measured the network's ability to recognize minor affine rotations and blurs of training images it had already memorized.

---

## 2. Corrected Leak-Free Architecture

To establish a scientifically valid and defensible machine learning workflow, the pipeline has been redesigned to enforce strict separation between data partitioning and data augmentation.

### 2.1 Corrected Workflow Sequence

```text
[ Raw Original Dataset ]
 (Unaugmented patient scans)
         │
         ▼
[ Ingestion, Verification & SHA-256 Hashing ]
 - Validates file readability
 - Rejects corrupt/empty images
 - Flags exact duplicate files
         │
         ▼
[ Stratified Train / Val / Test Partitioning ]
 - Executed on ORIGINAL images BEFORE any augmentation
 - Stratified by class: Train (70%), Val (15%), Test (15%)
         │
         ▼
[ Leakage & Contamination Safety Verification ]
 - Asserts: Train ∩ Val = ∅
 - Asserts: Train ∩ Test = ∅
 - Asserts: Val ∩ Test = ∅
         │
         ├───► [ Validation Set (15%) ] ──► Deterministic Preprocessing ──► Metric Evaluation
         │
         ├───► [ Test Set (15%) ] ────────► Deterministic Preprocessing ──► Final Benchmark
         │
         └───► [ Training Set (70%) ]
                     │
                     ▼
               [ Online On-the-Fly Augmentation ]
               (In-memory Keras layers during tf.data batching only)
                     │
                     ▼
               [ Model Training & Optimization ]
```

### 2.2 Key Architectural Improvements

| Aspect | Original Prototype | Corrected Pipeline |
| :--- | :--- | :--- |
| **Augmentation Timing** | Offline (written to disk before split) | **Online (applied on-the-fly during training batches)** |
| **Data Partitioning** | Random split on augmented pool (21,253 images) | **Stratified split on original images only (~253 images)** |
| **Disk Storage** | Generated 21,000+ files to disk | **Zero augmented files written to disk** |
| **Evaluation Data** | Contaminated with training image twins | **100% unseen, unaugmented original scans** |
| **Reproducibility** | Non-deterministic, unseeded OpenCV loops | **Deterministic random seed (`seed=42`)** |
| **Safety Auditing** | None | **Automated SHA-256 hash and path disjointness assertions** |

---

## 3. Dataset Constraints & Limitations

1. **Exact vs. Perceptual Duplication:**  
   The pipeline implements exact file matching via SHA-256 hashing. However, without clinical DICOM metadata (such as Patient ID, Series UID, Study Date), multiple slices originating from the same imaging volume cannot be programmatically grouped into patient clusters.
2. **Experimental Nature of Augmentations:**  
   The affine transformations (random flip, rotation, translation, zoom) are computer vision regularization techniques. They are **not** certified anatomical deformations.
3. **No Retraining Claims in Stage 2:**  
   In Stage 2, the pipeline infrastructure and splitting logic are established and verified. Retrained model performance will be evaluated on the uncompromised test set in Stage 3.
