# DSIM-USSL: Diffusion-based Seismic Inversion Model with Unsupervised-constrained Semi-Supervised Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **DSIM-USSL**, a novel deep learning framework for robust seismic impedance inversion proposed in the paper:

> **A diffusion-based robust seismic impedance inversion framework for enhanced hydrocarbon reservoir characterization**  

DSIM-USSL integrates a diffusion model with an unsupervised-constrained semi-supervised learning strategy to achieve high inversion accuracy and exceptional robustness under strong random noise. The method is validated on the Marmousi2 synthetic dataset and field seismic data from the Baiyun basin, South China Sea.

## 🌟 Key Features

- **Diffusion-based denoising** – Leverages the reverse Markov chain of a diffusion model to progressively remove noise during inversion.
- **Lightweight inversion module** – Combines CNN (local feature extraction) and GRU (global sequence modeling) for efficient spatiotemporal modeling.
- **Two-stage training (USSL)** – First, unsupervised pre‑training in a nearby area with similar geology; then, semi‑supervised fine‑tuning in the target area using limited well logs.
- **Strong noise robustness** – Outperforms conventional SSL, NSP‑SSL, and RSP‑SSL methods under extremely low SNR conditions (down to -5 dB).
- **Field‑data ready** – Successfully applied to real seismic data for reservoir characterization.

## 📁 Repository Structure

```
.
├── checkpoints/            # Saved model checkpoints
├── core/                   # Core modules (diffusion, inversion, losses)
├── data/                   # Dataset files (synthetic/field)
├── results_demultiple/     # Example outputs (e.g., denoised sections)
├── saved_models/           # Pre‑trained models
├── datasets.py             # Data loading utilities
├── datasets_2D.py          # 2D seismic data loader
├── forward_2D_models.py    # Forward modeling (CNN-based)
├── lossdata.npz            # Precomputed loss data (optional)
├── main.py                 # Main training/evaluation script
├── run.py                  # Simplified runner (example)
├── unet.py                 # U‑Net architecture for diffusion
├── utils.py                # Helper functions
└── visualization.py        # Plotting and display tools
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (with CUDA support recommended)
- Other dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm`

Install the required packages:

```bash
pip install torch numpy scipy matplotlib tqdm
```

### Data Preparation

- **Synthetic data (Marmousi2)**: The pre‑processed seismic and impedance data can be generated using the provided scripts (see `data/`). We include a small sample for quick testing.
- **Field data**: Due to confidentiality, the real dataset is not provided. Users can adapt the code to their own data by following the format described in the paper.

### Training

The framework involves three main steps, as described in Algorithms 1–3 of the paper.

#### 1. Train the diffusion model (U‑Net) in the original working area

```bash
python main.py --mode train_diffusion --data_path ./data/original_area --epochs 1000
```

#### 2. Unsupervised pre‑training of the inversion module in the original area

```bash
python main.py --mode pretrain_inversion --diffusion_ckpt ./checkpoints/diffusion.pth --epochs 20
```

#### 3. Semi‑supervised fine‑tuning in the target area

```bash
python main.py --mode finetune --pretrained_inv ./checkpoints/inv_pretrained.pth --data_path ./data/target_area --well_ratio 0.7
```

For a quick demonstration, run:

```bash
python run.py --example synthetic
```

### Evaluation

To evaluate a trained model on test data:

```bash
python main.py --mode test --ckpt ./checkpoints/finetuned.pth --data_path ./data/test --save_plots
```

Metrics (R², NRMSE, SNR) will be printed and saved.

## 📊 Results Summary

| Method          | Marmousi2 (Area 1, SNR = -5 dB) | Field data (Well W2) |
|-----------------|----------------------------------|----------------------|
| Conventional SSL| Poor continuity, strong artifacts| R² = 0.8231          |
| NSP‑SSL         | Moderate improvement            | R² = 0.8287          |
| RSP‑SSL         | Better than NSP, still unstable  | R² = 0.8594          |
| **DSIM‑USSL**   | **Clear reservoirs, sharp boundaries** | **R² = 0.8916**  |

For full quantitative and qualitative comparisons, please refer to the paper.


## 🤝 Contact

For questions or collaborations, please open an issue or contact the corresponding author at [yingtianliu06@outlook.com].

---

**Note**: The code is provided as-is for research purposes. We welcome contributions and feedback.
