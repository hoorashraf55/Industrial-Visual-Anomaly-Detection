# Individual Contribution Logs — AI403 Deep Learning Project

**Project Title:** AI-Powered Industrial Visual Anomaly Detection and Defect Localization  
**Dataset:** MVTec Anomaly Detection (MVTec AD)  
**University:** Pharos University in Alexandria | **Section:** D02  

---

## 1. Student Log: Khadija Mohamed Mohamed (ID: 202301025)
* **Assigned Role:** Classical Machine Learning Baseline (HOG + One-Class SVM)
* **Co-signed by:** Hoor Ashraf Khaled (ID: 202301368) — *Verified & Approved*

| Week | Task | Hours | Evidence (Commits, Runs, Report Sections) | Problems & Resolutions |
| :--- | :--- | :---: | :--- | :--- |
| **Week 1** | Literature review on classical industrial anomaly detection and setting up the baseline configuration (`configs/baseline.yaml`). | 6 | Commit: `init: add baseline yaml config`; Report Section 5 (Related Work) & Section 9 (Baseline). | Selecting appropriate HOG cell/block sizes to balance feature dimensionality and runtime on 256×256 RGB images. |
| **Week 2** | Implementing HOG feature extractor and One-Class SVM (`nu=0.05`, RBF kernel) in `src/models/baseline.py`. | 8 | Commit: `feat(baseline): implement HOG + OC-SVM pipeline`; Code module `src/models/baseline.py`. | High memory footprint when flattening raw HOG descriptors across all training images; resolved by standardizing feature scaling and batch extraction. |
| **Week 3** | Running multi-seed stability experiments on `bottle` (Seeds 42, 123, 2026) and full 15-category benchmark (Seed 42). | 9 | Runs: `experiments/results/baseline_*_seed42.json`; Report Tables 4A, 4B, 4C, and Table 7. | Weak pixel-level localization masks from patch-wise HOG grids; resolved by interpolating patch decision scores back to 256×256 resolution. |
| **Week 4** | Benchmarking computational cost (checkpoint size, training/inference time) and drafting Baseline methodology & comparison sections. | 6 | Report Sections 9, 13, 14, and Table 7 (`0.59 MB`, `9.72s` train time). | Ensuring uniform JSON output schema with deep learning models for automated table/figure generation. |

---

## 2. Student Log: Hoor Ashraf Khaled (ID: 202301368)
* **Assigned Role:** Project Coordinator, Repository Architecture & Data Pipeline
* **Co-signed by:** Hanen Tamer Ali (ID: 202303928) — *Verified & Approved*

| Week | Task | Hours | Evidence (Commits, Runs, Report Sections) | Problems & Resolutions |
| :--- | :--- | :---: | :--- | :--- |
| **Week 1** | Initializing GitHub repository structure, `.gitignore`, virtual environment (`PyTorch 2.11.0 + CUDA 13.0`), and MVTec AD download/extraction scripts. | 7 | Commits: `chore: setup repo structure and .gitignore`; `data/download.py`, `data/extract.py`; Report Section 6. | Preventing large `.pth` checkpoints and raw `.tar.xz` archives from being tracked by Git; resolved with strict `.gitignore` rules. |
| **Week 2** | Building unified PyTorch Dataset, DataLoader, preprocessing transforms (`256×256`), and leak-free train/validation split (`data/split.py`, `src/datasets/`). | 9 | Commits: `feat(data): add mvtec_dataset and leak-free val split`; Report Section 7 (Data Preprocessing). | Preventing data leakage during threshold calibration; resolved by splitting `train/good` into 80% training and 20% defect-free validation sets. |
| **Week 3** | Implementing optional data augmentation pipeline (`--augment`: horizontal flip & color jitter) and running the Data Strategy study on `bottle`. | 8 | Runs: `autoencoder_bottle_aug` vs `no_aug`; Report Section 13 (Study 2). | Augmentation slightly degraded pixel IoU (`0.1730` to `0.1191`) due to geometric shift sensitivity; documented in Discussion & Limitations. |
| **Week 4** | Coordinating final multi-seed/15-category runs, organizing Google Drive checkpoint links, and assembling `README.md` & Report Sections 1–7, 18–22. | 8 | Commit: `docs: finalize README and project deliverables`; Report Sections 1–7, 18–22. | Managing storage limits on institutional Google Drive; resolved by hosting shared checkpoint archives with public viewer permissions. |

---

## 3. Student Log: Hanen Tamer Ali (ID: 202303928)
* **Assigned Role:** CNN Autoencoder & Custom Reconstruction Loss
* **Co-signed by:** Mariam Wael Elnekidy (ID: 202303371) — *Verified & Approved*

| Week | Task | Hours | Evidence (Commits, Runs, Report Sections) | Problems & Resolutions |
| :--- | :--- | :---: | :--- | :--- |
| **Week 1** | Designing the symmetric Convolutional Autoencoder architecture (`2,430,915` params) and `configs/autoencoder.yaml`. | 7 | Commit: `feat(ae): add CNN autoencoder architecture`; `src/models/autoencoder.py`; Report Table 2. | Blurry reconstructions caused by standard pixel-wise MSE loss alone. |
| **Week 2** | Implementing `CombinedAnomalyLoss` ($\alpha \cdot \text{MSE} + (1-\alpha) \cdot (1-\text{SSIM})$, $\alpha=0.85$) in `src/components/custom_losses.py`. | 8 | Commit: `feat(loss): implement CombinedAnomalyLoss with SSIM`; Report Section 10.1. | Balancing structural similarity weight ($\alpha$) so small surface scratches are not smoothed out during training. |
| **Week 3** | Conducting hyperparameter studies (resolution `128` vs `256`, learning rate `0.0005` vs `0.001`), pretrained vs scratch encoder study, and 15-category runs. | 10 | Runs: `experiments/results/autoencoder_*`; Report Study 1, Study 3, and Tables 4A–4C. | False positives on stochastic textures (`leather.good.png`, `tile.good.png`) due to high-frequency texture variance; analyzed in Error Analysis. |
| **Week 4** | Evaluating inference speed (`35.50 ms` — fastest model) and writing Autoencoder methodology, hyperparameter, and error analysis sections. | 6 | Report Sections 10.1, 12, 16, and Table 6. | Low-contrast defects (`wood.scratch.png`, score `0.1942 < 0.2004`) falling just below threshold; documented as key limitation. |

---

## 4. Student Log: Mariam Wael Elnekidy (ID: 202303371)
* **Assigned Role:** PatchCore (Feature Memory Bank & Coreset Selection)
* **Co-signed by:** Reham Sayed Ezz Eldin (ID: 202301138) — *Verified & Approved*

| Week | Task | Hours | Evidence (Commits, Runs, Report Sections) | Problems & Resolutions |
| :--- | :--- | :---: | :--- | :--- |
| **Week 1** | Implementing multi-layer feature extraction (`layer2` & `layer3` from pretrained WideResNet-50) in `src/components/feature_extraction.py`. | 7 | Commit: `feat(patchcore): add multi-layer feature extractor`; Report Section 10.2. | Spatial resolution mismatch between `layer2` and `layer3` feature maps; resolved via bilinear upsampling and channel concatenation. |
| **Week 2** | Developing `MemoryBank` (`src/components/memory_bank.py`) and `greedy_coreset_selection` (`src/components/coreset.py`) to sub-sample nominal patch embeddings. | 9 | Commit: `feat(patchcore): implement greedy coreset memory bank`; Report Table 2. | Out-of-memory (OOM) errors on 4 GB RTX 3050 GPU when computing full pairwise distance matrices; resolved via random projection & batched k-NN search. |
| **Week 3** | Running 15-category benchmark (`0.9648` mean Image AUROC), Bottle multi-seed runs (`1.0000 ± 0.0000`), and resolution scaling study (`256` vs `320`). | 10 | Runs: `experiments/results/patchcore_*`; Report Tables 4A–4C and Study 1 (`0.5572` Pixel IoU at `320×320`). | Separating checkpoint metadata (`5.86 MB`) from memory bank serialization (`5.86 MB`) for clean loading in Streamlit. |
| **Week 4** | Benchmarking PatchCore runtime (`18.29s` training, `61.41 ms` inference) and authoring PatchCore technical sections & discussion. | 6 | Report Sections 10.2, 14, 17, and Table 7. | Explaining per-image min-max heatmap normalization on normal samples (`good.bottle.png`) for clear interpretation in the report. |

---

## 5. Student Log: Reham Sayed Ezz Eldin (ID: 202301138)
* **Assigned Role:** EfficientAD (Student-Teacher PDN & Ablation Study)
* **Co-signed by:** Menna Mohamed Makram (ID: 202303532) — *Verified & Approved*

| Week | Task | Hours | Evidence (Commits, Runs, Report Sections) | Problems & Resolutions |
| :--- | :--- | :---: | :--- | :--- |
| **Week 1** | Studying EfficientAD architecture and implementing the lightweight Patch Description Network (PDN) student-teacher backbone (`3,315,264` params). | 8 | Commit: `feat(efficientad): implement PDN student-teacher networks`; `src/models/efficientad.py`. | Aligning receptive field sizes between the teacher feature extractor and student PDN output channels. |
| **Week 2** | Integrating the global Autoencoder branch (`--no_ae` flag support) to capture both local structural and global logical anomalies. | 8 | Commit: `feat(efficientad): add global AE branch and ablation flag`; Report Section 10.3. | Training instability in early epochs due to scale differences between local ST loss and global AE discrepancy loss; stabilized with normalized feature targets. |
| **Week 3** | Executing 15-category benchmark (`0.1926` highest mean Pixel IoU), Bottle multi-seed runs (`0.9849 ± 0.0016`), and the Ablation Study (with vs without AE branch). | 10 | Runs: `experiments/results/efficientad_*`; Report Tables 4A–4C and Table 5 (Ablation Study). | Longer training time (`259.11s` per category on RTX 3050) compared to PatchCore; optimized dataloader workers and batch size (`4`). |
| **Week 4** | Analyzing ablation trade-offs (Image AUROC drop `0.9865 -> 0.9778` vs Pixel IoU gain `0.3766 -> 0.3968` & faster inference) and writing Sections 10.3 & 15. | 6 | Report Section 15 (Ablation Study) & Table 5 interpretation. | Investigating false negatives on low-contrast scratches (`wood.scratch.png`, `0.8966 < 0.9361`) due to conservative `95th percentile` validation threshold. |

---

## 6. Student Log: Menna Mohamed Makram (ID: 202303532)
* **Assigned Role:** Unified Evaluation Pipeline, Automated Reporting & Streamlit Live Demo
* **Co-signed by:** Khadija Mohamed Mohamed (ID: 202301025) — *Verified & Approved*

| Week | Task | Hours | Evidence (Commits, Runs, Report Sections) | Problems & Resolutions |
| :--- | :--- | :---: | :--- | :--- |
| **Week 1** | Designing the unified evaluation module (`src/evaluate.py`, `src/components/anomaly_scoring.py`) computing Image/Pixel AUROC, Precision, Recall, F1, and Pixel IoU. | 7 | Commit: `feat(eval): implement unified image and pixel metrics`; Report Section 11. | Ensuring zero threshold leakage by calibrating decision thresholds strictly on the defect-free validation split (`95th percentile`). |
| **Week 2** | Building `src/generate_report_assets.py` to automatically aggregate JSON results into Markdown tables (`report_tables.md`) and comparison figures (`experiments/figures/`). | 8 | Commit: `feat(reporting): add automated table and chart generator`; Figures 2–5 in Report. | Formatting standard deviations cleanly to distinguish across-seed SD (Table 4A) from across-category SD (Table 4B). |
| **Week 3** | Developing the interactive Streamlit web application (`demo/app.py`) with auto-calibrated thresholds, heatmap generation, and defect localization overlays. | 10 | Commit: `feat(demo): build interactive Streamlit inspection app`; `demo/app.py`. | Checkpoint key mismatches when switching dynamically between PatchCore, EfficientAD, and Autoencoder; resolved with unified model loader wrappers. |
| **Week 4** | Testing 40+ normal/anomalous samples across categories in Streamlit, capturing qualitative proofs, and compiling Table 6 & Section 16 (Error Analysis). | 7 | Report Table 6, Section 16, and Figures 6–29 (Streamlit & Error Analysis proofs). | Resolving matplotlib colormap scaling artifacts on defect-free images while keeping threshold indicators accurate. |