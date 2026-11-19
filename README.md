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

## 2. Team Members

| Student Name | Student ID | Responsibility |
|---|---|---|
| Khadija Mohamed Mohamed | 202301025 | Baseline (One-Class SVM) |
| Hoor Ashraf Khaled | 202301368 | Project Coordinator and Data Pipeline |
| Hanen Tamer Ali | 202303928 | CNN Autoencoder |
| Mariam Wael Elnekidy | 202303371 | PatchCore |
| Reham Sayed Ezz Eldin | 202301138 | EfficientAD |
| Menna Mohamed Makram | 202303532 | Evaluation and Streamlit Demo |

## 3. Dataset

### 3.1 Dataset Information

- **Dataset:** MVTec Anomaly Detection (MVTec AD)
- **Dataset Version:** Original MVTec AD (15 categories), not MVTec AD 2.
- **Official Source:** https://www.mvtec.com/research-teaching/datasets/mvtec-ad

The dataset contains:

- 5,354 high-resolution images.
- 15 object and texture categories.
- Defect-free training images.
- Normal and anomalous test images.
- Pixel-level ground-truth masks for anomalous test images.

### 3.2 Categories

The dataset includes the following 15 categories:

bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, and zipper.

### 3.3 Dataset Location

The training script uses the following default dataset path:

```text
C:\Users\Dell\Downloads\mvtec_anomaly_detection
```

If the dataset is stored in another location, provide the correct path using the `--data_root` argument.

The raw dataset is not included in the Git repository.

### 3.4 Dataset Download and Preparation

1. Download the original MVTec AD dataset from the official source.
2. Extract the downloaded archive.
3. Place the extracted dataset in a local directory.
4. Ensure that the category folders and their training, testing, and ground-truth directories are available.

For example, the extracted dataset should follow this structure:

```text
mvtec_anomaly_detection/
│
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   ├── broken_small/
│   │   └── contamination/
│   └── ground_truth/
│       ├── broken_large/
│       ├── broken_small/
│       └── contamination/
│
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
```

### 3.5 Dataset Preparation Commands

The current project does not use separate command-line scripts for downloading or extracting the dataset.

Download and extract the dataset from the official MVTec source, then configure its location using `--data_root`.

For example, in PowerShell:

```powershell
$env:MVTEC_ROOT = "D:\Datasets\mvtec_anomaly_detection"

Test-Path "$env:MVTEC_ROOT\bottle\train\good"
```

The command should return `True` if the dataset directory exists.

### 3.6 Data Preprocessing

Data preprocessing is performed automatically by the project's dataset loader:

`src/datasets/mvtec_dataset.py`

A separate preprocessing command is not required.

The implemented preprocessing steps include:

- Loading images as RGB.
- Resizing images to the selected input resolution (256 × 256 by default).
- Converting images to PyTorch tensors.
- Normalizing images using ImageNet mean and standard deviation.
- Resizing ground-truth masks using nearest-neighbor interpolation.
- Converting ground-truth masks into binary masks.

The training data is split into training and validation subsets using only defect-free images from `train/good`.

The default validation ratio is 20%, and the split is reproducible using the selected random seed.

The official test directory is kept separate and contains normal and anomalous images.

Data augmentation can optionally be enabled during training using the `--augment` argument. It is applied only to the training subset.

## 4. Environment Setup

### 4.1 Requirements

- Python 3.12
- PyTorch 2.11.0
- Torchvision 0.26.0
- CUDA 13.0
- NVIDIA GPU recommended

### 4.2 Create and Activate the Virtual Environment

Run the following commands from the project root directory:

```powershell
py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 4.3 Install PyTorch with CUDA Support

```powershell
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### 4.4 Install Project Dependencies

```powershell
pip install -r requirements.txt
```

### 4.5 Verify the Installation

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

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
│   │
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

## 6. Training

The project uses a shared training pipeline for all four models.

Run commands from the project root directory after activating the virtual environment.

If the dataset is stored in a custom directory, add:

```powershell
--data_root "D:\Datasets\mvtec_anomaly_detection"
```

to the training commands.

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

The training script supports the following arguments:

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

To use a pretrained Autoencoder encoder:

```powershell
python -m src.train --model autoencoder --category bottle --seed 42 --pretrained
```

To disable the EfficientAD autoencoder branch:

```powershell
python -m src.train --model efficientad --category bottle --seed 42 --no_ae
```

To enable training augmentation:

```powershell
python -m src.train --model patchcore --category bottle --seed 42 --augment
```

### 6.7 Multi-Seed Experiment

The multi-seed experiment uses the following seeds:

```powershell
python -m src.train --model autoencoder --category bottle --seed 42

python -m src.train --model autoencoder --category bottle --seed 123

python -m src.train --model autoencoder --category bottle --seed 2026
```

The same seed values were used across all 15 MVTec AD categories for all four models.

**Reproducibility Note:**

The three-seed stability experiment (seeds 42, 123, 2026) was executed across all 15 MVTec AD categories for all four models (180 core runs + 7 ablation, hyperparameter, and data-strategy runs = 187 total experiments).

## 7. Evaluation and Metrics

### 7.1 Evaluation Process

Evaluation is integrated into the training pipeline.

After each model finishes training, the pipeline evaluates it using the validation and test loaders and calculates the corresponding image-level and pixel-level metrics.

The evaluation process uses validation data to determine anomaly decision thresholds. The test set is used for final evaluation.

The evaluation results are automatically saved as JSON files under:

```text
experiments/results/
```

Each result file contains:

- Model name.
- Category.
- Random seed.
- Input image size.
- Experiment configuration.
- Image-level evaluation metrics.
- Pixel-level evaluation metrics.
- Training time.
- Inference time.
- Hardware information.

**Note:** The current `src/evaluate.py` file contains evaluation functions used by the training pipeline. It does not provide an independent command-line evaluation interface. Therefore, evaluation is executed automatically through the training command.

### 7.2 Evaluation Metrics

The evaluation includes the following metrics:

**Image-Level Metrics**

- Image AUROC
- Precision
- Recall
- F1-Score

**Pixel-Level Metrics**

- Pixel AUROC
- Pixel IoU

### 7.3 Generate Report Tables and Figures

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

## 8. Streamlit Demo

The project includes an interactive Streamlit application for visual inspection.

### 8.1 Run the Demo

After training the required models and ensuring that their checkpoints are available, launch the application using:

```powershell
streamlit run demo/app.py
```

### 8.2 Demo Features

The interface allows users to:

- Select a product category.
- Select a supported deep-learning model.
- Upload a product image.
- Adjust the anomaly decision threshold.
- View the inspection result.
- View the predicted anomaly map.
- View the localization overlay.

### 8.3 Supported Models

The demo supports:

- CNN Autoencoder
- PatchCore
- EfficientAD

The demo requires the corresponding trained checkpoints to be available in the configured checkpoints directory.

## 9. Experimental Results

The main benchmark evaluates all 15 MVTec AD categories across three random seeds (42, 123, 2026), with Seed 42 per-category comparisons reported below.

### 9.1 Main Model Comparison

| Model | Image AUROC | F1-Score | Pixel AUROC | Pixel IoU |
|---|---:|---:|---:|---:|
| HOG + One-Class SVM | 0.6776 | 0.3942 | 0.6796 | 0.0804 |
| CNN Autoencoder | 0.7757 | 0.5267 | 0.8220 | 0.1016 |
| PatchCore | 0.9648 | 0.9327 | 0.9730 | 0.1668 |
| EfficientAD | 0.8509 | 0.7185 | 0.9314 | 0.1926 |

### 9.2 Best Results

- PatchCore achieved the highest mean Image AUROC, F1-score, and Pixel AUROC across the 15 categories.
- EfficientAD achieved the highest mean Pixel IoU.
- PatchCore demonstrated strong image-level anomaly detection performance.
- EfficientAD achieved better average localization overlap in the reported benchmark.

These results are based on the evaluation across all 15 MVTec AD categories.

## 10. Hardware and Computational Cost

### 10.1 Hardware Used

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **GPU Memory:** 4 GB
- **Python:** 3.12.5
- **PyTorch:** 2.11.0
- **CUDA:** 13.0
- **Operating System:** Windows 11

### 10.2 Reported Computational Cost (Bottle Multi-Seed Experiment)

| Model | Parameters | Checkpoint / Model Size (MB) | Memory Bank (MB) | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline | N/A | 0.590 | N/A | 9.72 | 61.32 |
| Autoencoder | 2,430,915 | 9.30 | N/A | 95.50 | 35.50 |
| PatchCore | 24,862,528 | 5.86 | 5.86 | 18.29 | 61.41 |
| EfficientAD | 3,315,264 | 12.68 | N/A | 259.11 | 62.04 |

**Note:** These measurements are from the reported Bottle experiment and should not be interpreted as fixed runtimes for every category or hardware configuration.

Actual runtimes may vary depending on:

- Dataset category.
- GPU specifications.
- Batch size.
- Input image resolution.
- Available system resources.

Peak system RAM and GPU VRAM usage were not recorded.

PatchCore's memory-bank size is reported separately from its model/checkpoint size.

## 11. Experiments

The project includes the following experimental studies:

### 11.1 Hyperparameter Study

- Learning-rate comparison (lr = 0.0005 vs lr = 0.001).
- Image-resolution comparison (128x128, 256x256, and 320x320).

### 11.2 Data Strategy Study

- Training without augmentation.
- Training with augmentation.

### 11.3 Pretrained vs Scratch

- Autoencoder trained from scratch.
- Autoencoder using a pretrained encoder.

### 11.4 Ablation Study

- EfficientAD with its autoencoder branch.
- EfficientAD without its autoencoder branch.

### 11.5 Error Analysis

The project examines false positives and false negatives using representative normal and anomalous samples, along with anomaly maps and localization results.

The reported demo error-analysis examples are based on the displayed Streamlit cases and should not be interpreted as a complete evaluation of the entire MVTec AD test set.

## 12. Checkpoints and Large Files

Model checkpoints and the raw MVTec AD dataset are not included in the Git repository because of their size.

They should be stored in a shared drive folder, and the corresponding access link is provided below:

**Shared Drive Link:**

https://drive.google.com/drive/folders/1Hb7WIvnyWMhRCj_Ql9KuAfepi_DxeZH2?usp=drive_link

The `.gitignore` file excludes large checkpoint files and raw dataset folders from Git tracking.

## 13. Dataset Source and License

This project uses the original MVTec Anomaly Detection (MVTec AD) dataset, not MVTec AD 2.

The dataset was developed by MVTec Software GmbH for research on unsupervised industrial anomaly detection. It contains 5,354 high-resolution images across 15 object and texture categories, including defect-free training images, normal and anomalous test images, and pixel-level ground-truth masks.

### 13.1 Official Dataset Source

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### 13.2 Dataset License

The dataset is released under the:

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).**

License URL:

https://creativecommons.org/licenses/by-nc-sa/4.0/

### 13.3 Usage Conditions

- Attribution to the original dataset authors and source is required.
- The dataset must not be used for commercial purposes.
- Adaptations and redistributed material must comply with the ShareAlike requirements of the license.
- Users must review and comply with the original license terms before sharing or reusing the dataset.

The raw dataset is not redistributed as part of this GitHub repository. Users must obtain it from the official MVTec source and comply with its license.

### 13.4 Dataset Reference

Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019).

"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9584–9592.

https://doi.org/10.1109/CVPR.2019.00982

## 14. Model Sources, Implementations, and Pretrained Weights

The models in this project were implemented using PyTorch and TorchVision. Their architectural designs and anomaly detection strategies are based on the corresponding research literature.

The project-specific implementations are available in the `src/models/` directory.

### 14.1 HOG + One-Class SVM

The classical baseline combines Histogram of Oriented Gradients (HOG) feature extraction with a One-Class Support Vector Machine (OC-SVM) for anomaly detection.

**References:**

Dalal, N., and Triggs, B. (2005).

"Histograms of Oriented Gradients for Human Detection."

IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR).

Schölkopf, B., et al. (2001).

"Estimating the Support of a High-Dimensional Distribution."

Neural Computation, 13(7), 1443–1471.

### 14.2 CNN Autoencoder

The CNN Autoencoder is a project-specific reconstruction-based architecture implemented in PyTorch.

For the pretrained-versus-scratch experiment, the pretrained version uses a ResNet18 encoder initialized with the default pretrained weights provided by TorchVision:

```python
torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
)
```

The scratch version initializes its encoder without pretrained weights.

### 14.3 PatchCore

PatchCore is based on the method proposed in:

Roth, K., et al. (2022).

"Towards Total Recall in Industrial Anomaly Detection."

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

**Paper:**

https://arxiv.org/abs/2106.08265

The project implements a PatchCore-based approach using:

- A pretrained Wide ResNet-50-2 feature extraction backbone.
- Intermediate feature maps from the backbone.
- A memory bank of normal patch embeddings.
- Greedy coreset selection to reduce the memory bank.
- k-nearest-neighbor distance-based anomaly scoring.

The backbone is initialized using the default TorchVision pretrained weights:

```python
torchvision.models.wide_resnet50_2(
    weights=torchvision.models.Wide_ResNet50_2_Weights.DEFAULT
)
```

The backbone remains frozen during the experiment. The memory bank is constructed from the training data.

### 14.4 EfficientAD

EfficientAD is based on the student-teacher anomaly detection approach introduced in:

Batzner, K., Heckler, L., and König, R. (2024).

"EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies."

Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 128–138.

**Paper:**

https://arxiv.org/abs/2303.14535

The project implements an EfficientAD-inspired architecture containing:

- A frozen pretrained ResNet18 teacher feature extractor.
- A trainable student network based on a Patch Description Network (PDN)-style architecture.
- An optional autoencoder branch.
- Feature discrepancy-based anomaly maps.

The teacher network is initialized using TorchVision's default pretrained ResNet18 weights:

```python
torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT
)
```

The student network and the optional autoencoder branch are trained within the project training pipeline.

### 14.5 Pretrained Weights and Implementation Notes

Pretrained classification weights are provided by TorchVision and are automatically downloaded and cached when the corresponding model is initialized for the first time, if they are not already available locally.

**Official TorchVision documentation:**

https://docs.pytorch.org/vision/stable/models.html

The pretrained backbones used in this project are:

| Model | Backbone | Pretrained Weights Source |
|---|---|---|
| CNN Autoencoder (pretrained experiment) | ResNet18 | TorchVision |
| PatchCore | Wide ResNet-50-2 | TorchVision |
| EfficientAD teacher | ResNet18 | TorchVision |

The project does not claim to use pretrained PatchCore or EfficientAD experiment checkpoints from their original repositories. Instead, it uses the research methods as references and implements the corresponding project-specific pipelines in PyTorch.

The source code for these implementations is available in:

- `src/models/autoencoder.py`
- `src/models/patchcore.py`
- `src/models/efficientad.py`
- `src/training/`

## 15. External Code and Libraries

The project uses the following major libraries:

- PyTorch
- Torchvision
- NumPy
- OpenCV
- Scikit-learn
- Streamlit
- PyYAML
- Pillow

Any external code or implementation reused in the project should be credited with its original repository or publication.

## 16. Limitations

- Some subtle or low-contrast defects may be difficult to detect.
- Reconstruction-based methods may produce false positives on naturally irregular textures.
- Computational measurements depend on the available hardware.
- Peak RAM and VRAM usage were not recorded.
- The Streamlit demo requires the corresponding trained checkpoints.
- The project-specific implementations may differ from the complete original implementations described in the cited research papers.

## 17. Project Deliverables

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

## 18. How to Run the Project – Quick Start

Run the following commands from the project root directory.

### Step 1. Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2. Set the Dataset Location

```powershell
$env:MVTEC_ROOT = "D:\Datasets\mvtec_anomaly_detection"
```

### Step 3. Train a Model

```powershell
python -m src.train --model patchcore --category bottle --seed 42 --data_root "$env:MVTEC_ROOT"
```

### Step 4. Generate Report Tables and Figures

```powershell
python -m src.generate_report_assets
```

### Step 5. Launch the Demo

```powershell
streamlit run demo/app.py
```

**Evaluation Note:** Evaluation is performed automatically as part of the training pipeline. The resulting metrics are saved under `experiments/results/`.

## 19. References

1. Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019). "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR, pp. 9584–9592. https://doi.org/10.1109/CVPR.2019.00982

2. Roth, K., et al. (2022). "Towards Total Recall in Industrial Anomaly Detection." CVPR. https://arxiv.org/abs/2106.08265

3. Batzner, K., Heckler, L., and König, R. (2024). "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies." WACV. https://arxiv.org/abs/2303.14535

4. He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Deep Residual Learning for Image Recognition." CVPR, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

5. Dalal, N., and Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection." CVPR, pp. 886–893. https://doi.org/10.1109/CVPR.2005.177

6. Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson, R. C. (2001). "Estimating the Support of a High-Dimensional Distribution." Neural Computation, 13(7), 1443–1471. https://doi.org/10.1162/089976601750264965

7. TorchVision. "Models and Pretrained Weights." https://docs.pytorch.org/vision/stable/models.html

8. PyTorch. "PyTorch Documentation." https://pytorch.org/docs/stable/index.html

9. Scikit-learn. "One-Class SVM." https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

10. Streamlit. "Streamlit Documentation." https://docs.streamlit.io/