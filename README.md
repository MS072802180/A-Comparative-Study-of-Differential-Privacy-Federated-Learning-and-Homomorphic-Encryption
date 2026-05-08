# Privacy-Preserving Modulation Classification for Spectrum Sharing

A comparative study of Differential Privacy, Federated Learning, and Homomorphic Encryption on the RadioML 2018 dataset.

## Author

MS072802180

## Overview

This project implements and evaluates three privacy-preserving machine learning techniques for modulation classification:
- **Differential Privacy (DP)** using Opacus
- **Federated Learning (FL)** using custom FedAvg implementation
- **Homomorphic Encryption (HE)** using TenSEAL (CKKS scheme)

The baseline model is a 3-layer CNN achieving 91.12% accuracy on a 16,384-sample subset of RadioML 2018.01A (4 modulation classes, 10 dB SNR).

## Repository Structure
privacy_modulation_project/
│
├── data/ # Dataset files
│ ├── RADIOML 2018.hdf5 # Full dataset (20GB)
│ └── small_dataset.pkl # 16,384-sample subset
│
├── results/ # Experiment outputs
│ ├── baseline_accuracy.txt # 91.12%
│ ├── baseline_model.pth # Trained baseline CNN
│ ├── dp_results.txt # DP: epsilon vs accuracy
│ ├── fl_results.txt # FL: clients vs accuracy
│ ├── he_comparison_results.txt # HE: linear vs non-linear
│ └── final_all_results.txt # Complete summary
│
├── plots/ # Figures
│ ├── baseline_results.png
│ ├── dp_results.png
│ ├── fl_results.png
│ ├── he_comparison_results.png
│ └── technique_comparison_bar.png
│
├── figures/ # Copy of plots for paper
│
├── archive/ # Deprecated scripts (optional)
│
└── requirements.txt # Python dependencies


## Key Results

| Technique | Accuracy | Overhead |
|-----------|----------|----------|
| Baseline CNN | 91.12% | - |
| DP (ε=2.12) | 76.0% | ~2x training time |
| FL (10 clients) | 90.4% | O(c·d) per round |
| HE Linear | 28.5% | 75.8 ms inference |
| HE Non-linear (x²) | 69.0% | 1,516 ms inference |

## Scripts

### Setup and Exploration
- `explore_data.py` – Inspect HDF5 dataset structure
- `create_small_dataset.py` – Extract 16,384-sample subset (4 classes, 10 dB SNR)

### Baseline
- `baseline_fixed.py` – Train baseline CNN, save model and accuracy

### Differential Privacy
- `differential_privacy_fixed.py` – Run DP-SGD with noise multipliers 0.5, 1.0, 2.0, 5.0, 10.0

### Federated Learning
- `fl_simple.py` – Custom FedAvg implementation with 2, 5, 10, 20 clients

### Homomorphic Encryption
- `he_comparison.py` – Compare linear (logistic) vs non-linear (MLP with x²) HE models
- `he_scale_fixed.py` – Logistic regression HE with polynomial modulus 16384
- `he_mlp_squared_full.py` – MLP with x² activation on 8,000/500 train/test split

### Final Summary
- `final_all_results.py` – Generate unified plots and comparison tables

## Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
requirements.txt
text
torch>=1.12.0
torchvision
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
h5py>=3.0.0
opacus>=1.0.0
tenseal>=0.3.0
flwr>=1.0.0
jupyter
Note: flwr is listed but the working FL implementation uses custom FedAvg (not Flower simulation).

Dataset
The RadioML 2018.01A dataset can be downloaded from:

Kaggle: https://www.kaggle.com/datasets/pinxau1000/radioml2018

DeepSig: https://www.deepsig.ai/datasets

Place the downloaded RADIOML 2018.hdf5 file in the data/ folder.

Running the Experiments
Typical execution order:

bash
# 1. Explore and prepare data
python explore_data.py
python create_small_dataset.py

# 2. Baseline
python baseline_fixed.py

# 3. Differential Privacy (15-20 minutes)
python differential_privacy_fixed.py

# 4. Federated Learning (10-15 minutes)
python fl_simple.py

# 5. Homomorphic Encryption (30-60 minutes)
python he_comparison.py

# 6. Final summary
python final_all_results.py

Notes
HE experiments use subsets (4,000-8,000 train samples) due to computational constraints

x² activation is used instead of ReLU because ReLU causes CKKS scale overflow

BatchNorm layers are replaced with GroupNorm for DP compatibility

The Flower simulation backend failed on Windows; a custom FedAvg implementation is used instead

Citation
If using this code or results, please cite the RadioML dataset and the relevant papers listed in the References section of the accompanying paper.

License
This project is for academic research use. The RadioML dataset is under CC BY-NC-SA 4.0
