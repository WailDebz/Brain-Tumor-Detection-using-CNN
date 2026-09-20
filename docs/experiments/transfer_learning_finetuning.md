# Experiment Note: Controlled MobileNetV2 Fine-Tuning

**Experiment Identifier:** `transfer_mobilenetv2_finetuned`  
**Date:** September 2026  
**Status:** Completed (Validation-Only Development Stage)  
**Hardware Target:** CPU (Native Windows)  
**Dataset:** 228 unique image files (87 No Tumor, 141 Tumor) after exact SHA-256 deduplication  

---

## 1. Experimental Motivation

In Stage 4A, feature extraction with a completely frozen MobileNetV2 backbone achieved a best validation loss of $0.2487$ and validation accuracy of $91.18\%$ (Epoch 17). 

The goal of this experiment was to investigate whether controlled, conservative fine-tuning of the uppermost convolutional layers on the 159 training samples leads to further reduction in validation loss or improved representation of MRI-specific textures.

---

## 2. Test Set Independence & Non-Usage Disclosure

> **Critical Methodological Notice:**  
> The held-out 35-image test partition was **NOT accessed, evaluated, or inspected** during this experiment. Because the test set was previously used in Stage 4A as an exploratory model-comparison benchmark, it must not be used iteratively for model selection or hyperparameter tuning. All model development and observational comparisons in this stage rely strictly on the validation partition.

---

## 3. Architecture & Controlled Unfreezing Strategy

### 3.1 Unfrozen Layers vs. Frozen Lower Layers

MobileNetV2 contains 154 layers organized into 16 inverted residual blocks. To avoid catastrophic forgetting on a small dataset (159 training images), we applied a conservative unfreezing boundary:

- **Frozen Lower Backbone (Layers 0 to 142):** Blocks 1 through 15 remain completely frozen ($1,378,944$ non-trainable parameters).
- **Unfrozen Top Backbone (Layers 143 to 153):**
  - `block_16_expand` (Conv2D: $160 \to 960$ channels)
  - `block_16_depthwise` (DepthwiseConv2D: $3 \times 3$)
  - `block_16_project` (Conv2D: $960 \to 320$ channels)
  - `Conv_1` (Conv2D: $320 \to 1280$ channels)
  - Trainable backbone parameters: **879,040**
- **Trainable Classification Head:**
  - `GlobalAveragePooling2D()` $\to (1280,)$
  - `Dense(64, ReLU, L2=1e-4)`: 81,984 parameters
  - `Dropout(0.3)`
  - `Dense(1, Sigmoid)`: 65 parameters
  - Trainable head parameters: **82,049**
- **Total Parameters:** **2,340,033**
- **Total Trainable Parameters:** **961,089** ($41.1\%$ of model)
- **Total Non-Trainable Parameters:** **1,378,944** ($58.9\%$ of model)

### 3.2 BatchNormalization Treatment

When fine-tuning on small cohorts, BatchNormalization layers can destabilize training if mini-batch statistics are updated on small batches ($N = 32$).  
**Implementation Rule:** All `BatchNormalization` layers across the entire backbone (both frozen and unfrozen blocks) remain explicitly non-trainable (`layer.trainable = False`), locking their running mean and variance to the pretrained ImageNet distribution.

### 3.3 Learning Rate

A reduced learning rate of **$\text{lr} = 10^{-5}$** (100× smaller than the $10^{-3}$ rate used in feature extraction) was employed to prevent large gradient updates from destroying learned representations in the top convolutional block.

---

## 4. Training Execution & Observable Dynamics

- **Warm-Start Weights:** Initialized from the trained feature extractor (`models/brain_tumor_mobilenetv2_frozen.keras`).
- **Batch Size:** 32 (5 steps per epoch)
- **Maximum Epochs:** 30
- **Early Stopping:** `patience=8`, `restore_best_weights=True` monitored on `val_loss`.
- **LR Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)`.

```text
Epoch   1/30 - loss: 0.1671 - accuracy: 0.9245 - val_loss: 0.2650 - val_accuracy: 0.9118 [val_loss improved -> saved]
Epoch   2/30 - loss: 0.1577 - accuracy: 0.9308 - val_loss: 0.3066 - val_accuracy: 0.8824
Epoch   3/30 - loss: 0.1026 - accuracy: 0.9748 - val_loss: 0.3126 - val_accuracy: 0.8824
Epoch   4/30 - loss: 0.1821 - accuracy: 0.9497 - val_loss: 0.2753 - val_accuracy: 0.8824
Epoch   5/30 - loss: 0.1141 - accuracy: 0.9560 - val_loss: 0.2891 - val_accuracy: 0.8529 [lr -> 5e-6]
Epoch   6/30 - loss: 0.1322 - accuracy: 0.9560 - val_loss: 0.2625 - val_accuracy: 0.9118 [val_loss improved -> saved]
Epoch   7/30 - loss: 0.1362 - accuracy: 0.9497 - val_loss: 0.2511 - val_accuracy: 0.9118 [BEST EPOCH -> saved]
Epoch   8/30 - loss: 0.1283 - accuracy: 0.9560 - val_loss: 0.2776 - val_accuracy: 0.8824
Epoch   9/30 - loss: 0.1739 - accuracy: 0.9057 - val_loss: 0.2878 - val_accuracy: 0.8529
Epoch  10/30 - loss: 0.1439 - accuracy: 0.9371 - val_loss: 0.2976 - val_accuracy: 0.8529
Epoch  11/30 - loss: 0.1336 - accuracy: 0.9623 - val_loss: 0.2790 - val_accuracy: 0.9118 [lr -> 2.5e-6]
Epoch  12/30 - loss: 0.1105 - accuracy: 0.9748 - val_loss: 0.2749 - val_accuracy: 0.8824
Epoch  13/30 - loss: 0.1308 - accuracy: 0.9434 - val_loss: 0.2665 - val_accuracy: 0.8824
Epoch  14/30 - loss: 0.1236 - accuracy: 0.9686 - val_loss: 0.2534 - val_accuracy: 0.9118
Epoch  15/30 - loss: 0.1254 - accuracy: 0.9623 - val_loss: 0.2616 - val_accuracy: 0.8824 [lr -> 1.25e-6]
Epoch  15: Early stopping triggered. Restored weights from Epoch 7.
```

---

## 5. Validation-Only Development Comparison

The table below presents observable validation performance across all three developed model configurations:

| Model Configuration | Total Parameters | Trainable Parameters | Non-Trainable Parameters | Initial Learning Rate | Stopping Epoch | Best Validation Epoch | Best Validation Loss | Best Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN Baseline** | 101,889 | 101,441 | 448 | $10^{-3}$ | 13 | 5 | **0.6780** | **61.76%** (21/34) |
| **MobileNetV2 (Frozen Feature Extractor)** | 2,340,033 | 82,049 | 2,257,984 | $10^{-3}$ | 25 | 17 | **0.2487** | **91.18%** (31/34) |
| **MobileNetV2 (Controlled Fine-Tuning)** | 2,340,033 | 961,089 | 1,378,944 | $10^{-5}$ | 15 | 7 | **0.2511** | **91.18%** (31/34) |

---

## 6. Overfitting & Behavioral Observations

1. **Validation Performance Parity:**  
   Fine-tuning reached an optimal validation loss of $0.2511$ and validation accuracy of $91.18\%$ at Epoch 7, which is virtually identical to the frozen feature extractor ($0.2487$ loss, $91.18\%$ accuracy at Epoch 17).
2. **Training Loss Trajectory:**  
   Training loss decreased from $0.1671$ to $0.1026$ (reaching $97.48\%$ training accuracy by Epoch 3), while validation loss remained in the range $[0.2511, 0.3126]$. Early stopping correctly halted training at Epoch 15 to prevent further divergence.
3. **Diminishing Returns on Small Cohorts:**  
   On this 159-image training split, unfreezing 879,040 additional backbone parameters did not yield substantial validation loss improvements over the simpler 82,049-parameter frozen feature extractor.

---

## 7. Limitations & Context

- **Validation Cohort Size ($N = 34$):** The validation split contains 34 image files (13 No Tumor, 21 Tumor). A single misclassified image corresponds to a $2.94\%$ shift in accuracy.
- **Image-Level Deduplication:** Deduplication was performed via SHA-256 hashing. Because the source Kaggle dataset lacks DICOM header metadata (Patient IDs, Series UIDs), patient-level independence cannot be established from the metadata.
- **Non-Diagnostic Prototype:** This software is an experimental computer-vision benchmark and is not certified for clinical diagnostic use.
