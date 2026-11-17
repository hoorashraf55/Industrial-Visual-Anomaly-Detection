# MVTec Anomaly Detection (MVTec AD) Dataset Documentation

## 1. Dataset Information

- **Dataset Name:** MVTec Anomaly Detection (MVTec AD)
- **Official Source:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Citation:** Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Scale & Structure:** 5,354 high-resolution RGB images (approximately 700×700 to 1024×1024 pixels) across 15 industrial categories:
  - 10 object categories.
  - 5 texture categories.

## 2. Dataset Structure

The dataset follows the official MVTec AD directory structure:

- `train/good/`: Contains defect-free images used for unsupervised model training and validation.
- `test/`: Contains both normal (good) and anomalous images for final evaluation.
- `ground_truth/`: Contains pixel-level binary masks identifying defective regions in anomalous test images.

## 3. Leakage-Free Split Strategy

To ensure a leakage-free and reproducible evaluation, the following splitting strategy is applied:

### Training Set (80% of train/good)

- Selected exclusively from the official `train/good/` directory.
- Used only for unsupervised model fitting.
- Contains normal (defect-free) images only.

### Validation Set (20% of train/good)

- Selected exclusively from the remaining images in `train/good/`.
- Kept separate from the training set.
- Used for threshold calibration, specifically using the 95th percentile of normal validation anomaly scores.
- Does not contain anomalous samples.

### Test Set (test/)

- Uses the official MVTec AD test split without modification.
- Completely isolated from training and validation.
- Contains both defect-free (good) and anomalous samples.
- Used only for final model evaluation.

### Ground Truth Masks (ground_truth/)

- Official pixel-level binary masks are used to identify defective regions.
- Used for Pixel Localization evaluation.
- Support pixel-level metrics such as Pixel AUROC and segmentation-based evaluation metrics.

## 4. Reproducibility and Data Leakage Prevention

- Training and validation images are selected only from `train/good/`.
- The official `test/` directory remains completely isolated until final evaluation.
- No test images or ground-truth masks are used during model training or threshold calibration.
- Fixed random seeds (42, 123, 2026) are used to ensure reproducible data splitting.
- Each seed generates a corresponding training and validation split.
- The split information is saved in `data/split_manifest.json` to ensure consistent and reproducible experiments.

## 5. Summary

The MVTec AD dataset provides 15 industrial categories for evaluating unsupervised visual anomaly detection.

The proposed evaluation strategy ensures:
1. Training uses only normal images from `train/good/`.
2. Validation uses held-out normal images for threshold calibration.
3. Testing uses the official, isolated `test/` split.
4. Ground-truth masks support pixel-level anomaly localization evaluation.
5. Fixed random seeds and a split manifest ensure reproducibility.

This strategy prevents data leakage and provides a consistent evaluation framework for comparing anomaly detection models.
