# Generated Tables for AI403 Technical Report
## AI-Powered Industrial Visual Anomaly Detection and Defect Localization
### Dataset: MVTec Anomaly Detection (MVTec AD)

**Evaluation note:** The 15-category benchmark uses Seed 42. The three-seed stability experiment was conducted on the Bottle category using Seeds 42, 123, and 2026. Therefore, the mean ± standard deviation across seeds below applies to Bottle only, not to all 15 categories.

---

## Table 1. Dataset Comparison

| Dataset | Source | Size | Format | Classes | License / Usage | Role |
|---|---|---:|---|---:|---|---|
| MVTec AD | [Official MVTec AD Website](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 5,354 high-resolution images | RGB images and pixel-level ground-truth masks | 15 categories | Use according to the official MVTec dataset terms | Training and testing |

**Categories:** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, and zipper.

The dataset provides defect-free training images and test images containing both normal and anomalous samples. Pixel-level ground-truth masks are available for anomalous test images.

---

## Table 2. Model Architecture Comparison

| Model | Model Family | Input | Main Architecture / Components | Parameters | Pretrained |
|---|---|---|---|---:|---|
| HOG + One-Class SVM | Classical Machine Learning Baseline | 256 × 256 RGB images | HOG feature extraction followed by One-Class SVM | N/A | No |
| CNN Autoencoder | Reconstruction-based Deep Learning | 256 × 256 RGB images | CNN encoder-decoder with reconstruction-based anomaly scoring | 2,430,915 | No, in the main configuration |
| PatchCore | Feature-based Anomaly Detection | 256 × 256 RGB images | Pretrained ResNet feature extraction, multi-layer feature aggregation, memory bank, coreset sampling, and nearest-neighbor distance scoring | 24,862,528 | Yes |
| EfficientAD | Student-Teacher Anomaly Detection | 256 × 256 RGB images | Custom PDN-based student-teacher architecture with an autoencoder branch and anomaly-map scoring | 3,315,264 | Teacher uses pretrained features |

**Additional implementation details:**
- The Autoencoder uses a custom combined anomaly loss.
- PatchCore uses a memory bank and a custom greedy coreset selection component.
- EfficientAD includes a custom PDN component and an autoencoder branch.
- The EfficientAD ablation experiment disables the autoencoder branch.

---

## Table 3. Hyperparameter Configuration

| Model / Experiment | Optimizer / Method | Learning Rate | Batch Size | Epochs | Regularization | Loss / Scoring | Augmentation |
|---|---|---:|---:|---:|---|---|---|
| HOG + One-Class SVM | One-Class SVM | N/A | N/A | N/A | nu = 0.05 | HOG features and One-Class SVM anomaly scoring | No |
| CNN Autoencoder | Adam | 0.001 | 4 | 15 | Weight decay = 1e-5 | CombinedAnomalyLoss, alpha = 0.85 | Optional |
| PatchCore | Feature extraction and memory-bank construction | N/A | 4 | N/A | Coreset sampling | Nearest-neighbor feature distance | Optional |
| EfficientAD | Adam | 0.001 | 4 | 15 | Model-specific | Student-teacher discrepancy with AE branch | Optional |
| EfficientAD Ablation | Adam | 0.001 | 4 | 15 | Model-specific | Student-teacher discrepancy without AE branch | Optional |

**Common experimental settings:**
- Default image size: 256 × 256 pixels.
- Default batch size: 4.
- Default number of epochs for trainable deep-learning models: 15.
- Default learning rate: 0.001.
- Main benchmark seed: 42.
- Bottle multi-seed experiment: 42, 123, and 2026.

**Augmentation:** The training pipeline supports optional augmentation, including random horizontal flipping and color jitter. Augmentation is enabled through the `--augment` training argument.

---

## Table 4A. Multi-Seed Stability on Bottle (Mean ± Seed SD)

| Model | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU | Parameters | Checkpoint Size (MB) | Train Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.6894 ± 0.0151 | 0.3196 ± 0.0823 | 0.5869 ± 0.0038 | 0.0511 ± 0.0018 | N/A | 0.59 | 9.72 | 61.32 |
| Autoencoder | 0.8971 ± 0.0057 | 0.8493 ± 0.0069 | 0.8011 ± 0.0113 | 0.1578 ± 0.0210 | 2,430,915 | 9.30 | 95.50 | 35.50 |
| PatchCore | 1.0000 ± 0.0000 | 0.9922 ± 0.0134 | 0.9869 ± 0.0002 | 0.2286 ± 0.0049 | 24,862,528 | 5.86 | 18.29 | 61.41 |
| EfficientAD | 0.9849 ± 0.0016 | 0.9676 ± 0.0223 | 0.9768 ± 0.0011 | 0.3794 ± 0.0059 | 3,315,264 | 12.68 | 259.11 | 62.04 |

*The values in this table summarize the three-seed Bottle experiment. Training and inference times are the reported averages for that experiment.*

---

## Table 4B. Overall Benchmark Performance Across All 15 MVTec AD Categories (Seed 42)

| Model | Categories | Image AUROC (Mean ± Category SD) | Mean Precision | Mean Recall | F1-Score (Mean ± Category SD) | Pixel AUROC (Mean ± Category SD) | Pixel IoU (Mean ± Category SD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 15 | 0.6776 ± 0.1222 | 0.9127 | 0.2733 | 0.3942 ± 0.1940 | 0.6796 ± 0.1976 | 0.0804 ± 0.1107 |
| Autoencoder | 15 | 0.7757 ± 0.1806 | 0.9036 | 0.4344 | 0.5267 ± 0.2927 | 0.8220 ± 0.1224 | 0.1016 ± 0.0657 |
| PatchCore | 15 | 0.9648 ± 0.0504 | 0.9503 | 0.9267 | 0.9327 ± 0.0814 | 0.9730 ± 0.0207 | 0.1668 ± 0.1332 |
| EfficientAD | 15 | 0.8509 ± 0.1148 | 0.9280 | 0.6365 | 0.7185 ± 0.2073 | 0.9314 ± 0.0626 | 0.1926 ± 0.1055 |

**Main observation:** PatchCore achieved the highest mean Image AUROC, F1-score, and Pixel AUROC across the 15 categories. EfficientAD achieved the highest mean Pixel IoU, indicating stronger average localization overlap in this benchmark.

**Important:** The standard deviation in Table 4B is across categories, not across random seeds.

---

## Table 4C. Per-Class Image AUROC Across All 15 Categories (Seed 42)

| Category | Baseline | Autoencoder | PatchCore | EfficientAD |
|---|---:|---:|---:|---:|
| bottle | 0.7040 | 0.9008 | 1.0000 | 0.9865 |
| cable | 0.4959 | 0.5708 | 0.9856 | 0.7701 |
| capsule | 0.6462 | 0.6677 | 0.9465 | 0.8145 |
| carpet | 0.6091 | 0.5951 | 0.9852 | 0.8832 |
| grid | 0.6608 | 0.9841 | 0.9557 | 0.5514 |
| hazelnut | 0.7193 | 0.9721 | 1.0000 | 0.9714 |
| leather | 0.6332 | 0.8876 | 1.0000 | 0.9783 |
| metal_nut | 0.6491 | 0.3856 | 0.9990 | 0.9692 |
| pill | 0.5783 | 0.7608 | 0.9514 | 0.8200 |
| screw | 0.5091 | 0.8576 | 0.8123 | 0.7383 |
| tile | 0.6079 | 0.7204 | 1.0000 | 0.7886 |
| toothbrush | 0.8917 | 0.9472 | 0.9028 | 0.9056 |
| transistor | 0.7550 | 0.6008 | 0.9862 | 0.8542 |
| wood | 0.9184 | 0.9667 | 0.9851 | 0.8956 |
| zipper | 0.7862 | 0.8178 | 0.9617 | 0.8369 |
| **Mean** | **0.6776** | **0.7757** | **0.9648** | **0.8509** |

---

## Table 5. Ablation Study: EfficientAD With AE Branch vs Without AE Branch

| Configuration | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| EfficientAD with AE branch | 0.9865 | 0.9841 | 0.9759 | 0.3766 | 255.13 | 60.84 |
| EfficientAD without AE branch | 0.9778 | 0.9760 | 0.9759 | 0.3968 | 187.71 | 52.88 |

**Interpretation:**

Removing the autoencoder branch reduced Image AUROC from 0.9865 to 0.9778 and F1-score from 0.9841 to 0.9760. Pixel AUROC remained almost unchanged, while Pixel IoU increased from 0.3766 to 0.3968. The ablated model also required less training time and had lower inference time.

This indicates that the autoencoder branch contributed to image-level detection performance, while removing it improved localization overlap and computational efficiency in this experiment.

---

## Table 6. Error Analysis

The following examples are based on the 19 displayed Streamlit demo cases. They represent observed demo examples, not a complete statistical evaluation of all 44 reviewed images or the entire test set.

| Model | Error Type | Count in the 19 Displayed Cases | Example | Likely Cause |
|---|---|---:|---|---|
| Autoencoder | False Positive | 1 | tile.good.png | Natural irregular texture may produce reconstruction differences that exceed the anomaly threshold. |
| Autoencoder | False Negative | 1 | wood.scratch.png | The subtle scratch score (0.1942) remained below the threshold (0.2004). |
| EfficientAD | False Negative | 1 | wood.scratch.png | The low-contrast scratch was not sufficiently distinguished from normal wood texture; score (0.8966) remained below threshold (0.9361). |
| PatchCore | No classification error in the displayed examples | 0 | Displayed normal and anomalous examples | The displayed samples were correctly classified; this does not establish zero errors on the complete test set. |

**Error Analysis Discussion**

The error analysis revealed different failure patterns across the evaluated models. The Autoencoder produced a false positive on the normal `tile.good.png` sample. This may be caused by the difficulty of reconstructing stochastic textures accurately. Small reconstruction differences across naturally irregular patterns can increase the reconstruction error and cause a normal image to exceed the anomaly threshold.

The Autoencoder also produced a false negative on `wood.scratch.png`. Although the anomaly map showed a response around the scratch, the image-level anomaly score (0.1942) remained slightly below the threshold (0.2004). This indicates that a subtle defect may be localized in the anomaly map without producing a sufficiently high image-level score.

EfficientAD also produced a false negative on the same `wood.scratch.png` sample. The low-contrast scratch was not sufficiently distinguished from the normal wood texture, and the image-level score (0.8966) remained below the threshold (0.9361).

In the displayed examples, PatchCore correctly classified the shown normal and anomalous samples, including the bent metal nut and scratched wood. However, these examples alone are not sufficient to establish a zero-error rate across the complete test set.

Overall, the observed failures show that subtle defects and naturally irregular textures remain challenging cases, particularly for reconstruction-based anomaly detection.

---

## Table 7. Computational Cost

| Model | Parameters | Checkpoint / Model Size (MB) | Memory Bank (MB) | Training Time (s) | Inference Time (ms) | Hardware |
|---|---:|---:|---:|---:|---:|---|
| HOG + One-Class SVM | N/A | 0.59 | 0 | 9.72 | 61.32 | NVIDIA GeForce RTX 3050 Laptop GPU |
| CNN Autoencoder | 2,430,915 | 9.30 | 0 | 95.50 | 35.50 | NVIDIA GeForce RTX 3050 Laptop GPU |
| PatchCore | 24,862,528 | 5.86 | 5.86 | 18.29 | 61.41 | NVIDIA GeForce RTX 3050 Laptop GPU |
| EfficientAD | 3,315,264 | 12.68 | 0 | 259.11 | 62.04 | NVIDIA GeForce RTX 3050 Laptop GPU |

**Memory note:** The reported PatchCore memory-bank size is separate from the checkpoint/model size. Peak system RAM and GPU VRAM usage were not recorded in the available experiment logs and are therefore not reported.

**Computational observation:** PatchCore achieved the strongest overall image-level benchmark results while using a compact reported checkpoint size and a separate memory bank. EfficientAD achieved the highest mean Pixel IoU but required more training time in the reported Bottle multi-seed experiment.

---

## Additional Experimental Studies

### Study 1. Hyperparameter Study: Learning Rate and Image Resolution

| Experiment | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Autoencoder, image size 128 | 0.9698 | 0.8224 | 0.8324 | 0.1895 | 79.39 | 34.47 |
| Autoencoder, image size 256 | 0.9008 | 0.8468 | 0.8028 | 0.1730 | 93.94 | 35.58 |
| Autoencoder, learning rate 0.0005 | 0.9008 | 0.8468 | 0.8028 | 0.1730 | 93.94 | 35.58 |
| Autoencoder, learning rate 0.001 | 0.8976 | 0.8113 | 0.7890 | 0.1427 | 88.90 | 32.43 |
| PatchCore, image size 256 | 1.0000 | 1.0000 | 0.9869 | 0.2320 | 18.08 | 58.25 |
| PatchCore, image size 320 | 1.0000 | 0.9921 | 0.9870 | 0.5572 | 20.37 | 75.58 |

### Study 2. Data Strategy: Without Augmentation vs With Augmentation

| Configuration | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Autoencoder without augmentation | 0.9008 | 0.8468 | 0.8028 | 0.1730 | 93.94 | 35.58 |
| Autoencoder with augmentation | 0.8889 | 0.8148 | 0.7381 | 0.1191 | 112.57 | 44.76 |

### Study 3. Pretrained vs Scratch Encoder (Autoencoder)

| Configuration | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Autoencoder trained from scratch | 0.9008 | 0.8468 | 0.8028 | 0.1730 | 93.94 | 35.58 |
| Autoencoder with pretrained encoder | 0.9349 | 0.8649 | 0.8160 | 0.1863 | 126.45 | 39.10 |

---

## Reproducibility and Experimental Limitations

1. The main benchmark covers all 15 MVTec AD categories using Seed 42.
2. Three random seeds (42, 123, and 2026) were recorded for the Bottle multi-seed experiment.
3. The three-seed experiment was not completed for every category and every final model configuration. Therefore, the Bottle seed variance must not be interpreted as 15-category multi-seed variance.
4. The standard deviation in the overall 15-category table is calculated across categories.
5. The error-analysis counts refer only to the displayed demo examples.
6. Peak RAM and VRAM usage were not recorded.
7. All reported computational-cost measurements were collected on an NVIDIA GeForce RTX 3050 Laptop GPU.