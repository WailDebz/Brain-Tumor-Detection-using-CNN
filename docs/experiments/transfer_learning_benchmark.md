# Experiment Note: Transfer Learning Benchmark (MobileNetV2 Feature Extractor)

**Experiment Identifier:** `transfer_mobilenetv2_frozen`  
**Date:** September 2026  
**Status:** Completed (Feature Extraction Stage)  
**Hardware Target:** CPU (Native Windows)  
**Dataset:** 228 unique image files (87 No Tumor, 141 Tumor) after exact SHA-256 deduplication  

---

## 1. Experimental Motivation

In Stage 3C, a custom 3-block 2D CNN trained from scratch on 159 training samples achieved $\text{ROC-AUC} = 0.7902$, but its predicted probabilities were concentrated in a narrow upper interval ($[0.52, 0.85]$).

Small medical imaging datasets ($N \approx 200\text{--}300$) often present optimization challenges when training all convolutional filters from scratch on small sample counts. The objective of this experiment is to evaluate whether representations transferred from an ImageNet-pretrained backbone (MobileNetV2) can serve as a fixed feature extractor when evaluated under the exact same leak-free data partition.

---

## 2. Test Set Usage Disclosure & Experimental Protocol

> **Important Methodological Disclosure:**  
> The held-out 35-image split was used as an exploratory benchmark for comparing the corrected custom CNN and MobileNetV2 feature-extraction models. Because architecture comparison was informed by this benchmark, these results should not be interpreted as an untouched final estimate of generalization.
>
> **Future Experimental Policy:** Model development, architecture exploration, and hyperparameter selection decisions must rely exclusively on training/validation procedures. The existing 35-image test partition must not be used for iterative hyperparameter tuning.

---

## 3. Model Architecture & Input Adaptation

### 3.1 Backbone Selection: MobileNetV2

- **Backbone Architecture:** MobileNetV2 (Sandler et al., 2018) with inverted residual blocks and depthwise separable convolutions.
- **Pretrained Weights:** ImageNet-1k (`include_top=False`).
- **Rationale:** 
  - Parameter efficiency: 2,257,984 backbone parameters (~8.6 MB).
  - Practical CPU training and inference latency on standard workstations.
  - Linear bottleneck structure preserves manifold representations across layer depth.

### 3.2 In-Graph Input Adaptation & Normalization

The data pipeline standardizes single-channel grayscale axial MRI slices to $(90, 90, 1)$ in $[0.0, 1.0]$. To interface with MobileNetV2 without modifying raw data files:
1. **Channel Expansion:** An in-graph `Concatenate(axis=-1, name="grayscale_to_rgb")([x, x, x])` layer replicates the single grayscale channel across 3 RGB channels $\to (90, 90, 3)$.
2. **Dynamic Range Rescaling:** A `Rescaling(scale=2.0, offset=-1.0)` layer maps $[0.0, 1.0]$ to $[-1.0, +1.0]$, matching MobileNetV2's expected ImageNet input range.
3. **Frozen BatchNorm Verification:** The backbone is invoked with `training=False` during feature extraction, ensuring BatchNormalization running statistics remain frozen to their ImageNet values during training.

### 3.3 Parameter Breakdown

- **Backbone Parameters (Non-trainable):** 2,257,984
- **Classification Head (Trainable):**
  - `GlobalAveragePooling2D()`: 0 parameters $\to (1280,)$
  - `Dense(64, ReLU, L2=1e-4)`: $(1280 \times 64) + 64 = 81,984$ parameters
  - `Dropout(0.3)`: 0 parameters
  - `Dense(1, Sigmoid)`: $(64 \times 1) + 1 = 65$ parameters
- **Total Head Trainable Parameters:** **82,049**
- **Total Model Parameters:** **2,340,033** (82,049 trainable, 2,257,984 non-trainable)

---

## 4. Training Protocol & Leak-Free Safeguards

- **Data Partitions (Identical to Custom CNN Baseline):**
  - Training Set: **159 image files** ($69.7\%$) with online random augmentation
  - Validation Set: **34 image files** ($14.9\%$) deterministic, unaugmented
  - Held-Out Test Set: **35 image files** ($15.4\%$) deterministic, unaugmented
- **Optimizer:** Adam ($\text{lr} = 0.001$)
- **Loss:** Binary Crossentropy
- **Batch Size:** 32 (5 steps per epoch)
- **Early Stopping:** `patience=8`, `restore_best_weights=True` monitored on `val_loss`.
- **Learning Rate Schedule:** `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)`.
- **Test Set Quarantine:** The test partition was not accessed during training, early stopping, or threshold selection.

---

## 5. Observed Training Dynamics

```text
Epoch   1/30 - loss: 1.3583 - accuracy: 0.5094 - val_loss: 0.9190 - val_accuracy: 0.6471
Epoch   2/30 - loss: 0.6851 - accuracy: 0.6981 - val_loss: 0.5404 - val_accuracy: 0.7353
Epoch   3/30 - loss: 0.3762 - accuracy: 0.8302 - val_loss: 0.4643 - val_accuracy: 0.7941
Epoch   6/30 - loss: 0.3169 - accuracy: 0.8553 - val_loss: 0.4008 - val_accuracy: 0.8529
Epoch  10/30 - loss: 0.2756 - accuracy: 0.8679 - val_loss: 0.3075 - val_accuracy: 0.8824
Epoch  13/30 - loss: 0.2143 - accuracy: 0.8994 - val_loss: 0.2925 - val_accuracy: 0.8824
Epoch  17/30 - loss: 0.2108 - accuracy: 0.9371 - val_loss: 0.2487 - val_accuracy: 0.9118 [BEST EPOCH]
Epoch  21/30 - loss: 0.1984 - accuracy: 0.9560 - val_loss: 0.2737 - val_accuracy: 0.8824 [lr -> 0.0005]
Epoch  25/30 - loss: 0.1554 - accuracy: 0.9434 - val_loss: 0.2566 - val_accuracy: 0.8824 [lr -> 0.00025]
Epoch  25: Early stopping triggered. Restored weights from Epoch 17.
```

- **Best Validation Epoch:** 17
- **Best Validation Loss:** **0.2487** (vs. 0.6780 for Custom CNN)
- **Best Validation Accuracy:** **91.18%** (31/34 samples correctly classified)

---

## 6. Held-Out Test Evaluation Results ($N = 35$)

### 6.1 Primary Benchmark: Fixed Default Threshold ($\theta = 0.50$)

| Metric | MobileNetV2 (Frozen Backbone) | Custom CNN Baseline | Observed Difference |
| :--- | :---: | :---: | :---: |
| **Decision Threshold ($\theta$)** | **0.5000** | **0.5000** | — |
| **True Positives (TP)** | **19** | 22 | $-3$ |
| **True Negatives (TN)** | **11** | 0 | $+11$ |
| **False Positives (FP)** | **2** | 13 | $-11$ |
| **False Negatives (FN)** | **3** | 0 | $+3$ |
| **Accuracy** | **85.71%** | 62.86% | $+22.85\%$ |
| **Sensitivity (Recall / TPR)** | **86.36%** | 100.00% | $-13.64\%$ |
| **Specificity (TNR)** | **84.62%** | 0.00% | $+84.62\%$ |
| **Precision (PPV)** | **90.48%** | 62.86% | $+27.62\%$ |
| **F1-Score** | **88.37%** | 77.19% | $+11.18\%$ |
| **ROC-AUC** | **0.9231** | 0.7902 | $+0.1329$ |
| **PR-AUC (Average Precision)** | **0.9657** | 0.8089 | $+0.1568$ |

---

### 6.2 Secondary Benchmark: Validation-Tuned Threshold ($\theta^* = 0.8332$)

When the decision threshold was selected strictly on the Validation set via Youden's J ($\theta^* = 0.8332$) and applied once to the test set:
- $\text{TP} = 16, \text{TN} = 13, \text{FP} = 0, \text{FN} = 6$
- $\text{Accuracy} = 82.86\%$
- $\text{Sensitivity} = 72.73\%$
- $\text{Specificity} = 100.00\%$
- $\text{Precision} = 100.00\%$
- $\text{F1-Score} = 84.21\%$

---

## 7. Observational Findings & Analysis

1. **Probability Distribution & Threshold Behavior:**  
   The frozen MobileNetV2 model produced output probabilities spanning a wider numerical range ($[0.05, 0.98]$) than the custom CNN (whose outputs clustered in $[0.52, 0.85]$). This wider distribution allowed the fixed default 0.5 threshold to separate positive and negative classes without requiring threshold offset.
2. **Observed Discrimination Differences:**  
   In this benchmark, MobileNetV2 achieved a higher validation accuracy ($91.18\%$ vs. $61.76\%$), a lower validation loss ($0.2487$ vs. $0.6780$), and a higher test ROC-AUC ($0.9231$ vs. $0.7902$) than the custom CNN trained from scratch.

---

## 8. Limitations & Non-Clinical Context

- **Small Test Cohort ($N = 35$):** The exploratory test benchmark comprises 35 image files (13 No Tumor, 22 Tumor). Each sample corresponds to $\approx 2.86\%$ of accuracy.
- **Image-Level Deduplication:** Deduplication was performed at the file level via SHA-256 hashing. Because the source Kaggle dataset lacks DICOM headers and patient identifiers, patient-level independence cannot be established from the metadata.
- **Non-Diagnostic Prototype:** This software is an experimental computer-vision benchmark and is not certified for clinical diagnostic use.

---

## 9. Fine-Tuning Considerations

The frozen feature extraction experiment demonstrated lower validation loss ($0.2487$) and higher test ROC-AUC ($0.9231$) than the custom CNN. A subsequent fine-tuning experiment (unfreezing upper backbone layers with a small learning rate such as $10^{-5}$) is technically plausible to explore whether domain-specific feature adaptation alters performance. Any such fine-tuning must be developed strictly on the training/validation sets.

