# AI-Powered Industrial Visual Anomaly Detection and Defect Localization

**Course:** Deep Learning (AI403) – Fall 2026  
**University:** Pharos University in Alexandria  
**Section:** D02  
**Project Area:** Deep Learning / Computer Vision

---

## 1. Project Overview

This project presents an AI-powered industrial visual anomaly detection system using deep learning and computer vision.

The system analyzes product images to:

- Classify products as normal or anomalous.
- Detect visual defects such as scratches, cracks, dents, and other irregularities.
- Generate anomaly maps to highlight suspicious regions.
- Evaluate image-level detection and pixel-level localization performance.

The project compares four approaches:

1. HOG + One-Class SVM (Classical Machine Learning Baseline).
2. CNN Autoencoder.
3. PatchCore.
4. EfficientAD.

The experiments use the MVTec Anomaly Detection (MVTec AD) dataset.

---

## 2. Team Members

| Student Name | Student ID | Responsibility |
|---|---|---|
| Khadija Mohamed Mohamed | 202301025 | Baseline (One-Class SVM) |
| Hoor Ashraf Khaled | 202301368 | Project Coordinator and Data Pipeline |
| Hanen Tamer Ali | 202303928 | CNN Autoencoder |
| Mariam Wael Elnekidy | 202303371 | PatchCore |
| Reham Sayed Ezz Eldin | 202301138 | EfficientAD |
| Menna Mohamed Makram | 202303532 | Evaluation and Streamlit Demo |

---

## 3. Dataset

**Dataset:** MVTec Anomaly Detection (MVTec AD)

**Official Source:**  
https://www.mvtec.com/company/research/datasets/mvtec-ad

The dataset contains:

- 5,354 high-resolution images.
- 15 object and texture categories.
- Defect-free training images.
- Normal and anomalous test images.
- Pixel-level ground-truth masks for anomalous test images.

### Categories

`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, and `zipper`.

### Dataset Location

The training script uses the following default dataset path:

```text
C:\Users\Dell\Downloads\mvtec_anomaly_detection
```

If the dataset is stored in another location, provide the correct path using `--data_root`.

### Dataset Download and Preparation

1. Download MVTec AD from its official source.
2. Extract the downloaded archive.
3. Place the extracted dataset in the configured data directory.
4. Ensure that the category folders and their training, testing, and ground-truth directories are available.

The raw dataset is not included in the Git repository.

### Dataset Preparation Commands

Run the following commands from the project root directory to inspect the available dataset preparation options:

```powershell
python -m data.download --help
python -m data.extract --help
python -m data.preprocessing --help
```

After checking the supported arguments, execute the corresponding download, extraction, and preprocessing commands.

**Note:** The exact command-line arguments depend on the implementation of each preparation script.

---

## 4. Environment Setup

### Requirements

- Python 3.12
- PyTorch 2.11.0
- Torchvision 0.26.0
- CUDA 13.0
- NVIDIA GPU recommended

### Create and Activate the Virtual Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install PyTorch with CUDA Support

```powershell
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### Install Project Dependencies

```powershell
pip install -r requirements.txt
```

### Verify the Installation

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

---

## 5. Project Structure

```text
Industrial-Visual-Anomaly-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   ├── autoencoder.yaml
│   ├── patchcore.yaml
│   ├── efficientad.yaml
│   └── experiments.yaml
│
├── data/
│   ├── download.py
│   ├── extract.py
│   ├── dataset.py
│   ├── split.py
│   └── preprocessing.py
│
├── src/
│   ├── datasets/
│   │   ├── mvtec_dataset.py
│   │   ├── transforms.py
│   │   └── dataloaders.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── autoencoder.py
│   │   ├── patchcore.py
│   │   └── efficientad.py
│   │
│   ├── components/
│   │   ├── feature_extraction.py
│   │   ├── memory_bank.py
│   │   ├── coreset.py
│   │   ├── anomaly_scoring.py
│   │   └── custom_losses.py
│   │
│   ├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── generate_report_assets.py
│
├── experiments/
│   ├── results/
│   ├── figures/
│   └── report_tables.md
│
├── checkpoints/
│
├── demo/
│   └── app.py
│
├── report/
│
└── contributions/
```

---

## 6. Training

The project uses a shared training and evaluation pipeline for all four models.

Run commands from the project root directory after activating the virtual environment.

### 6.1 Baseline – HOG + One-Class SVM

```powershell
python -m src.train --model baseline --category bottle --seed 42
```

### 6.2 CNN Autoencoder

```powershell
python -m src.train --model autoencoder --category bottle --seed 42
```

### 6.3 PatchCore

```powershell
python -m src.train --model patchcore --category bottle --seed 42
```

### 6.4 EfficientAD

```powershell
python -m src.train --model efficientad --category bottle --seed 42
```

### 6.5 Train on Another Category

Replace `bottle` with any supported MVTec AD category.

For example:

```powershell
python -m src.train --model patchcore --category cable --seed 42
```

### 6.6 Custom Training Arguments

The training script supports:

| Argument | Description |
|---|---|
| `--model` | Model name: baseline, autoencoder, patchcore, or efficientad |
| `--category` | MVTec AD category |
| `--seed` | Random seed |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--image_size` | Input image resolution |
| `--lr` | Learning rate |
| `--data_root` | Dataset directory |
| `--config` | Custom YAML configuration |
| `--pretrained` | Use a pretrained Autoencoder encoder |
| `--no_ae` | Disable the EfficientAD autoencoder branch |
| `--augment` | Enable training augmentation |

Example:

```powershell
python -m src.train --model autoencoder --category bottle --seed 42 --epochs 15 --batch_size 4 --image_size 256 --lr 0.001
```

### 6.7 Multi-Seed Experiment

The Bottle multi-seed experiment uses:

```powershell
python -m src.train --model autoencoder --category bottle --seed 42
python -m src.train --model autoencoder --category bottle --seed 123
python -m src.train --model autoencoder --category bottle --seed 2026
```

The same seed values were used for the Bottle stability experiments with the other models.

**Reproducibility Note:** The three-seed experiment was performed on the Bottle category. The main 15-category benchmark uses Seed 42.

---

## 7. Evaluation and Metrics

The training pipeline evaluates each trained model and saves the experiment results as JSON files under:

```text
experiments/results/
```

### Evaluation Command

To inspect the available evaluation arguments:

```powershell
python -m src.evaluate --help
```

Use the supported arguments to evaluate a trained model independently.

**Note:** The training pipeline also performs evaluation automatically. The exact standalone evaluation command depends on the arguments implemented in `src/evaluate.py`.

### Evaluation Metrics

The evaluation includes:

**Image-Level Metrics**

- Image AUROC
- Precision
- Recall
- F1-Score

**Pixel-Level Metrics**

- Pixel AUROC
- Pixel IoU

The evaluation also uses validation data to determine anomaly decision thresholds.

### Generate Report Tables and Figures

After the experiment results have been generated, run:

```powershell
python -m src.generate_report_assets
```

This command generates or updates:

- `experiments/report_tables.md`
- `experiments/figures/`
- Overall model comparison charts.
- Per-class comparison charts.
- Error-analysis examples.

---

## 8. Streamlit Demo

The project includes an interactive Streamlit application for visual inspection.

### Run the Demo

```powershell
streamlit run demo/app.py
```

The interface allows users to:

1. Select a product category.
2. Select a supported deep-learning model.
3. Upload a product image.
4. Adjust the anomaly decision threshold.
5. View the inspection result.
6. View the predicted anomaly map.
7. View the localization overlay.

The demo supports:

- CNN Autoencoder
- PatchCore
- EfficientAD

The demo requires the corresponding trained checkpoints to be available in the configured checkpoints directory.

---

## 9. Experimental Results

The main benchmark evaluates all 15 MVTec AD categories using Seed 42.

| Model | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU |
|---|---:|---:|---:|---:|
| HOG + One-Class SVM | 0.6776 | 0.3942 | 0.6796 | 0.0804 |
| CNN Autoencoder | 0.7757 | 0.5267 | 0.8220 | 0.1016 |
| PatchCore | **0.9648** | **0.9327** | **0.9730** | 0.1668 |
| EfficientAD | 0.8509 | 0.7185 | 0.9314 | **0.1926** |

### Best Results

- PatchCore achieved the highest mean Image AUROC, F1-score, and Pixel AUROC across the 15 categories.
- EfficientAD achieved the highest mean Pixel IoU.
- PatchCore demonstrated strong image-level anomaly detection performance.
- EfficientAD achieved better average localization overlap in the reported benchmark.

These results are based on the Seed 42 evaluation across the 15 categories.

---

## 10. Hardware and Computational Cost

### Hardware Used

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU Memory: 4 GB
- Python: 3.12.5
- PyTorch: 2.11.0
- CUDA: 13.0
- Operating System: Windows 11

### Reported Computational Cost (Bottle Multi-Seed Experiment)

| Model | Parameters | Checkpoint / Model Size (MB) | Memory Bank (MB) | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline | N/A | 0.59 | 0 | 9.72 | 61.32 |
| Autoencoder | 2,430,915 | 9.30 | 0 | 95.50 | 35.50 |
| PatchCore | 24,862,528 | 5.86 | 5.86 | 18.29 | 61.41 |
| EfficientAD | 3,315,264 | 12.68 | 0 | 259.11 | 62.04 |

**Note:** These measurements are from the reported Bottle experiment and should not be interpreted as fixed runtimes for every category or hardware configuration.

Actual runtimes may vary depending on:

- Dataset category.
- GPU specifications.
- Batch size.
- Input image resolution.
- Available system resources.

Peak system RAM and GPU VRAM usage were not recorded.

PatchCore's memory-bank size is reported separately from its model/checkpoint size.

---

## 11. Experiments

The project includes the following experimental studies:

### Hyperparameter Study

- Learning-rate comparison.
- Image-resolution comparison.

### Data Strategy Study

- Training without augmentation.
- Training with augmentation.

### Pretrained vs Scratch

- Autoencoder trained from scratch.
- Autoencoder using a pretrained encoder.

### Ablation Study

- EfficientAD with its autoencoder branch.
- EfficientAD without its autoencoder branch.

### Error Analysis

The project examines false positives and false negatives using representative normal and anomalous samples, along with anomaly maps and localization results.

---

## 12. Checkpoints and Large Files

Model checkpoints and the raw MVTec AD dataset are not included in the Git repository because of their size.

They should be stored in a shared drive folder, and the corresponding access link should be added here:

**Shared Drive Link:** (https://drive.google.com/drive/folders/1Hb7WIvnyWMhRCj_Ql9KuAfepi_DxeZH2?usp=drive_link)

The `.gitignore` file excludes large checkpoint files and raw dataset folders from Git tracking.

---

## 13. Credits and References

### Dataset

Bergmann et al., MVTec Anomaly Detection Dataset (MVTec AD).

Official Source:  
https://www.mvtec.com/company/research/datasets/mvtec-ad

### Models and Methods

- **One-Class SVM:** Classical anomaly detection baseline.
- **CNN Autoencoder:** Reconstruction-based anomaly detection.
- **PatchCore:** Patch-based feature memory-bank anomaly detection.
- **EfficientAD:** Student-teacher-based anomaly detection.

External implementations, pretrained models, and research methods should be acknowledged and cited using their original publications and official sources.

### External Code and Libraries

The project uses the following major libraries:

- PyTorch
- Torchvision
- NumPy
- OpenCV
- Scikit-learn
- Streamlit

Any external code or implementation reused in the project should be credited with its original repository or publication.

---

## 14. Limitations

- The three-seed stability experiment was performed on the Bottle category rather than all 15 categories.
- Some subtle or low-contrast defects may be difficult to detect.
- Reconstruction-based methods may produce false positives on naturally irregular textures.
- Computational measurements depend on the available hardware.
- Peak RAM and VRAM usage were not recorded.
- The Streamlit demo requires the corresponding trained checkpoints.

---

## 15. Project Deliverables

The project repository includes:

- Organized source code.
- Dataset documentation.
- Model implementations.
- Training and evaluation pipeline.
- Experiment results.
- Generated report tables and figures.
- Streamlit demonstration application.
- Technical report and presentation.
- Individual contribution logs.

---

## 16. How to Run the Project – Quick Start

Run the following commands from the project root directory.

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Train a model
python -m src.train --model patchcore --category bottle --seed 42

# Generate report tables and figures
python -m src.generate_report_assets

# Launch the demo
streamlit run demo/app.py
```

For standalone evaluation, first inspect the supported arguments:

```powershell
python -m src.evaluate --help
```

For dataset preparation, inspect the supported commands:

```powershell
python -m data.download --help
python -m data.extract --help
python -m data.preprocessing --help
```

---

**End of README**