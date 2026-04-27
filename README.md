# DarkMatter-ML-Euclid 🌌

**Mapping the invisible universe with machine learning — inspired by the ESA Euclid mission.**

This repository implements a complete ML pipeline for dark-matter mass reconstruction from weak gravitational lensing data. It includes two complementary approaches:

1. **CNN Mass Mapper (U-Net)** — Reconstructs convergence (κ) maps from noisy shear fields (γ₁, γ₂).
2. **GNN Halo Mass Predictor** — Estimates dark-matter halo masses from galaxy point clouds using graph neural networks.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)

---

## Overview

Weak gravitational lensing — the subtle distortion of background galaxy shapes by foreground matter — is one of the most powerful probes of the dark matter distribution. The ESA Euclid mission will map the shapes of over a billion galaxies, enabling unprecedented constraints on cosmology.

This project explores ML-based approaches to two key analysis tasks:

| Task | Input | Output | Model |
|------|-------|--------|-------|
| Mass mapping | Noisy shear field (γ₁, γ₂) | Convergence map κ | U-Net (CNN) |
| Halo mass estimation | Galaxy point cloud | log₁₀(M_halo / M☉) | EdgeConv / GAT (GNN) |

---

## Architecture

### CNN: U-Net Mass Mapper

```
Input: (B, 2, H, W)  [γ₁, γ₂ shear channels]
  │
  ├── Encoder ──┐   (4 stages, each: Conv → ResBlock → MaxPool)
  │             │
  ├── Bottleneck│   (2 × ResBlock at lowest resolution)
  │             │
  ├── Decoder ──┘   (4 stages, each: UpConv → AttentionGate + Skip → ResBlock)
  │
Output: (B, 1, H, W)  [convergence κ]
```

Key features:
- Residual blocks with GroupNorm and GELU activations
- Attention gates on skip connections (focus on cluster regions)
- Combined loss: pixel MSE + power spectrum + peak count

### GNN: Halo Mass Predictor

```
Galaxy point cloud (N galaxies × 8 features)
  │
  ├── k-NN graph construction (k=16)
  │
  ├── Node encoder MLP
  │
  ├── 4 × Message Passing (EdgeConv or GATv2) + LayerNorm + residual
  │
  ├── Global pooling (mean ∥ max)
  │
  ├── Regression MLP → log₁₀(M_halo)
```

---

## Repository Structure

```
DarkMatter-ML-Euclid/
├── README.md                  # This file
├── PROGRESS.md                # Detailed research log (Mar 1 – Apr 27, 2026)
├── requirements.txt           # Python dependencies
├── train.py                   # Training pipeline (CNN & GNN)
├── evaluate.py                # Evaluation with cosmological metrics
├── data/
│   ├── __init__.py
│   └── mock_data_generator.py # Synthetic Euclid-like data generation
├── models/
│   ├── __init__.py
│   ├── cnn_mapper.py          # U-Net mass mapper
│   └── gnn_mapper.py          # GNN halo mass predictor
├── utils/
│   ├── __init__.py
│   └── metrics.py             # Cosmological evaluation metrics
└── tests/
    ├── __init__.py
    ├── test_data_generator.py # Tests for data pipeline
    ├── test_models.py         # Tests for CNN & GNN architectures
    └── test_metrics.py        # Tests for evaluation metrics
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Yash5852176/DarkMatter-ML-Euclid.git
cd DarkMatter-ML-Euclid

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** For GPU support, install PyTorch with CUDA first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

```bash
# 1. Generate synthetic data
python -m data.mock_data_generator --n-maps 200 --n-catalogs 100 --size 128 --out data/generated

# 2. Train the CNN mass mapper
python train.py cnn --epochs 50 --n-maps 200 --batch-size 8

# 3. Train the GNN halo mass predictor
python train.py gnn --epochs 30 --n-catalogs 200 --batch-size 16

# 4. Evaluate
python evaluate.py cnn --checkpoint checkpoints/cnn_best.pt --n-test 50
python evaluate.py gnn --checkpoint checkpoints/gnn_best.pt --n-test-catalogs 50
```

---

## Data Generation

The `mock_data_generator.py` creates synthetic weak-lensing data:

- **Convergence maps:** Log-normal random fields with a Limber-approximation power spectrum matching Euclid survey parameters (30 gal/arcmin², σ_ε = 0.26, z_s = 1.0).
- **Shear fields:** Derived from convergence via the Kaiser-Squires forward model.
- **Galaxy catalogs:** Halos with Tinker-like mass function, galaxies placed via NFW profiles, with mock observables (magnitude, colour, size, ellipticities).

---

## Training

### CNN
```bash
python train.py cnn \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --n-maps 500 \
    --map-size 128 \
    --base-features 64 \
    --depth 4
```

### GNN
```bash
python train.py gnn \
    --epochs 50 \
    --batch-size 32 \
    --lr 3e-4 \
    --n-catalogs 500 \
    --hidden-dim 128 \
    --gnn-layers 4 \
    --conv-type edge
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/
```

---

## Evaluation

### Metrics computed

| Category | Metrics |
|----------|---------|
| Pixel-level | MSE, MAE, Pearson r, SSIM |
| Fourier-space | Power spectrum ratio P_pred/P_true, cross-correlation r(ℓ) |
| Peak statistics | Peak counts above Nσ thresholds |
| GNN-specific | Scatter (dex), outlier fraction |

---

## Results

| Model | Metric | Value |
|-------|--------|-------|
| CNN (U-Net) | Pixel MSE | 9.5 × 10⁻⁶ |
| CNN (U-Net) | Pearson r | 0.89 |
| CNN (U-Net) | Power spectrum error (ℓ < 7000) | < 8% |
| CNN (U-Net) | Peak recovery (> 3σ) | 87% |
| GNN (EdgeConv) | Scatter | 0.33 dex |
| GNN (EdgeConv) | Pearson r | 0.79 |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{darkmatter_ml_euclid,
  title={DarkMatter-ML-Euclid: ML-based mass mapping for weak lensing},
  author={Yash},
  year={2026},
  url={https://github.com/Yash5852176/DarkMatter-ML-Euclid}
}
```

### Key references

- Jeffrey et al. (2020) — [DeepMass](https://arxiv.org/abs/2011.08271)
- Villanueva-Domingo et al. (2022) — [GNN halo masses](https://arxiv.org/abs/2111.14874)
- Kaiser & Squires (1993) — Original KS inversion
- Laureijs et al. (2011) — Euclid Definition Study

---

*Built with ❤️ for dark matter research.*
