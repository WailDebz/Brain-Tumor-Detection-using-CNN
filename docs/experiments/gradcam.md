# Visual Saliency & Explainability: Grad-CAM Implementation

**Module:** `src/explainability/gradcam.py`  
**Target Model:** MobileNetV2 Frozen Feature Extractor (`models/brain_tumor_mobilenetv2_frozen.keras`)  
**Target Layer:** `out_relu` / `Conv_1` (Shape: $3 \times 3 \times 1280$)  
**Status:** Completed (Independent Explainability Module)  

---

## 1. Overview & Methodological Background

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) is an interpretability technique for convolutional neural networks that generates visual heatmaps highlighting the spatial regions of an input image that influence a specific class score.

In this repository, Grad-CAM is implemented as a standalone, deterministic explainability module to provide qualitative inspection of visual features contributing to the model's output score.

---

## 2. Target Layer Selection & Graph Architecture

### 2.1 Selected Target Layer: `out_relu`

In the MobileNetV2 architecture, the target convolutional feature layer was selected based on the following criteria:
1. **Upstream of Global Pooling:** Located immediately prior to `GlobalAveragePooling2D()` and the classification head.
2. **Spatial Feature Retention:** Preserves spatial 2D feature dimensions ($3 \times 3 \times 1280$ channels at $90 \times 90$ input resolution).
3. **High-Level Semantic Abstraction:** `out_relu` represents the rectified output of the final convolutional projection layer (`Conv_1`), capturing the highest-level spatial representations prior to 1D pooling.

### 2.2 Mathematical Gradient Flow

For an input tensor $X$ and target layer activation tensor $A \in \mathbb{R}^{H_f \times W_f \times K}$ (where $H_f = 3, W_f = 3, K = 1280$):

1. **Target Score Formulation ($y^c$):**
   - For **Tumor (Positive Class, $c = 1$):** $y^{c=1} = p = \sigma(z)$ (the predicted tumor probability score).
   - For **No Tumor (Negative Class, $c = 0$):** $y^{c=0} = 1.0 - p$ (the predicted non-tumor probability score).
2. **Channel Importance Weights ($\alpha_k^c$):**  
   Computed via Global Average Pooling of gradients:
   $$\alpha_k^c = \frac{1}{H_f \cdot W_f} \sum_{i=1}^{H_f} \sum_{j=1}^{W_f} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Weighted Linear Combination & Rectification:**  
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{K} \alpha_k^c A^k \right)$$
   The $\text{ReLU}$ non-linearity ensures that only features positively contributing to class score $y^c$ are retained.
4. **Safe Normalization:**  
   $$L_{\text{norm}}^c = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c)}$$
   (If $\max(L^c) = 0$, an all-zero matrix is returned safely without division by zero or NaN values).
5. **Bilinear Upsampling:** $L_{\text{norm}}^c$ is interpolated from $(3 \times 3)$ to the original image dimensions $(W, H)$ and blended with the RGB scan:
   $$\text{Overlay} = \alpha \cdot \text{Colormap}(L_{\text{norm}}^c) + (1 - \alpha) \cdot I_{\text{RGB}}$$

---

## 3. Preprocessing Consistency

To ensure scientific validity, Grad-CAM utilizes the exact same deterministic preprocessing pipeline as normal model inference:
1. Input MRI image (PIL Image or NumPy array) $\to$ single-channel grayscale array.
2. Resized to $(90, 90)$ via area interpolation.
3. Scaled to $[0.0, 1.0]$ float32.
4. Replicated in-graph to $(90, 90, 3)$ and rescaled to $[-1.0, +1.0]$ for the MobileNetV2 backbone.

---

## 4. Binary Class Interpretation & Asymmetry

- **Positive Class ($c = 1$, Tumor):** Visualizes spatial regions whose feature activations increase the predicted probability of tumor presence.
- **Negative Class ($c = 0$, No Tumor):** Visualizes spatial regions whose feature activations decrease the predicted probability of tumor presence (increase $1 - p$).
- **Methodological Caveat on Binary Sigmoid Grad-CAM:**  
  Because $\frac{\partial (1-p)}{\partial A} = - \frac{\partial p}{\partial A}$, negative-class Grad-CAM reflects inverse gradient directions. Saliency maps for negative predictions highlight areas that suppress tumor confidence, which does not represent anatomical "healthy tissue segmentation."

---

## 5. Qualitative Demonstration Artifacts

Demonstration figures generated via `scripts/generate_gradcam_example.py` are saved in `reports/figures/gradcam/`:

- `gradcam_brain_tumor_transfer_mobilenetv2_Y1_tumor.png`: Demonstration on an axial scan with tumor ($p = 0.9994$).
- `gradcam_brain_tumor_transfer_mobilenetv2_1_no_no_tumor.png`: Demonstration on a non-tumor axial scan ($p = 0.0747$).

Each visualization contains:
1. Original input MRI scan.
2. Normalized $2\text{D}$ Grad-CAM heatmap ($[0.0, 1.0]$).
3. Superimposed saliency overlay ($\alpha = 0.45$).

---

## 6. Scientific & Clinical Limitations

1. **Qualitative Only:** Grad-CAM provides a qualitative visualization of regions contributing to the model output. It does **not** prove that the model understands tumor pathology or anatomy.
2. **Coarse Spatial Resolution:** Because the input resolution is $90 \times 90$ pixels, the final convolutional feature map resolution is only **$3 \times 3$ pixels**. Upsampling from $3 \times 3$ to $90 \times 90$ (or native image resolution) creates broad, diffuse heatmaps that cannot delineate precise anatomical boundaries or micro-lesion margins.
3. **No Clinical Ground-Truth Validation:** Saliency maps have not been validated against radiologist ground-truth bounding boxes or voxel-level segmentations.
4. **Non-Diagnostic Disclaimer:** This tool is an educational and exploratory research prototype. It is not certified for clinical diagnosis, patient triage, or treatment planning.
