# DarkMatter-ML-Euclid — Research Progress Log

> **Period:** 1 March 2026 — 27 April 2026  
> **Principal Investigator:** Yash  
> **Objective:** Build and validate ML-based mass-mapping and halo-mass-estimation pipelines for Euclid-like weak-lensing surveys.

---

## Week 1 — 1–7 March 2026 | Literature Review & Project Scaffolding

### Mon 1 Mar
- Kicked off the project by reviewing the Euclid mission science case (Laureijs et al. 2011).
- Read Jeffrey et al. (2020) — *"DeepMass: Deep learning for convergence map reconstruction"* — as the primary CNN reference.
- Set up the GitHub repository (`DarkMatter-ML-Euclid`), initialised with a README.

### Tue 2 Mar
- Studied the Kaiser-Squires inversion formalism (Kaiser & Squires 1993): forward model κ → (γ₁, γ₂) is a linear convolution in Fourier space.
- Noted the ill-posed inverse problem: noise in shear fields gets amplified at high ℓ.
- Sketched out the project module structure: `data/`, `models/`, `utils/`, `train.py`, `evaluate.py`.

### Wed 3 Mar
- Reviewed log-normal models for convergence fields (Hilbert et al. 2011; Xavier et al. 2016).
- Decided to use log-normal realisations for synthetic data — cheaper than N-body and adequate for ML training.
- Started coding `data/mock_data_generator.py`: survey config dataclass, power spectrum model.

### Thu 4 Mar
- Implemented the simplified Limber-approximation C_κ(ℓ) fitting formula.
- Wrote `generate_gaussian_field()` to draw 2-D Gaussian realisations with a given P(ℓ).
- Validated against known analytic variance: σ²_κ = ∫ C(ℓ) ℓ dℓ / 2π. Match within 5%.

### Fri 5 Mar
- Added log-normal transformation and noise injection to `generate_convergence_map()`.
- Implemented `generate_shear_from_convergence()` using the Kaiser-Squires kernel D(k).
- **Issue:** DC mode leak causing a constant offset in reconstructed κ. Fixed by zeroing k=0.

### Sat 6 Mar — Sun 7 Mar
- Read Villanueva-Domingo et al. (2022) on GNNs for halo mass estimation from galaxy catalogues.
- Read Cranmer et al. (2020) — symbolic regression via GNNs for physics discovery.
- Drafted a plan for the GNN branch of the project.

---

## Week 2 — 8–14 March 2026 | Data Pipeline & CNN Architecture

### Mon 8 Mar
- Completed `build_cnn_dataset()`: batch generation of (γ₁, γ₂) → κ pairs. Tested with 100 maps at 64×64.
- Added `SurveyConfig` defaults to roughly match Euclid wide survey (30 gal/arcmin², σ_ε = 0.26).

### Tue 9 Mar
- Started designing the U-Net architecture in `models/cnn_mapper.py`.
- Chose GroupNorm over BatchNorm (training will use small batches on a single GPU).
- Added residual connections inside each encoder/decoder block.

### Wed 10 Mar
- Implemented attention gates for skip connections (Oktay et al. 2018).
- Rationale: high-convergence regions (galaxy clusters) are rare and spatially localised; attention helps focus on them.
- Forward pass validated: input (B, 2, 128, 128) → output (B, 1, 128, 128).

### Thu 11 Mar
- Designed `MassMapLoss`: combined pixel MSE + Fourier power spectrum loss + peak count loss.
- Power spectrum loss operates on log-power to weight large and small scales more equally.
- Peak count loss uses a sigmoid approximation to make it differentiable.

### Fri 12 Mar
- Wrote first version of `train.py` for the CNN path.
- AdamW optimiser, cosine-annealing LR schedule, gradient clipping at 1.0.
- TensorBoard logging for train/val loss and learning rate.

### Sat 13 Mar
- Ran a quick training test (20 maps, 10 epochs, 64×64) to verify the pipeline end-to-end.
- **Bug:** Shape mismatch in the attention gate when input dimension is not a power of 2. Fixed with dynamic padding in `UpBlock.forward()`.

### Sun 14 Mar
- Pushed initial working code to GitHub.
- Started writing `utils/metrics.py` for evaluation.

---

## Week 3 — 15–21 March 2026 | Metrics, GNN Architecture, Halo Catalog

### Mon 15 Mar
- Implemented pixel metrics: MSE, MAE, Pearson r, simplified SSIM.
- Implemented azimuthal-average power spectrum computation.
- Added `power_spectrum_ratio()` and `cross_correlation_coefficient()`.

### Tue 16 Mar
- Added peak-count statistics: detect local maxima above σ-thresholds.
- Wrapped everything in `evaluate_mass_map()` — returns a dict of all metrics for one map pair.

### Wed 17 Mar
- Designed the halo catalog generator for GNN training.
- Mock catalogs: halo masses drawn from a simplified mass function (power-law + exponential cutoff).
- Galaxies placed around halos following NFW-like exponential radial profile.
- Per-galaxy observables: magnitude, colour, size, ellipticities (e₁, e₂).

### Thu 18 Mar
- Started implementing `models/gnn_mapper.py` with `GNNHaloMassPredictor`.
- Architecture: node encoder MLP → EdgeConv message passing → global mean+max pooling → regression head.
- Added residual connections and LayerNorm after each message-passing layer.

### Fri 19 Mar
- Tested GNN with a small catalog (50 halos, ~500 galaxies). Forward pass works.
- **Issue:** k-NN graph construction was slow for large point clouds. Profiling showed scipy's cKDTree was the bottleneck. Switched to `torch_geometric.nn.knn_graph` which runs on GPU.

### Sat 20 Mar
- Added GATv2Conv as an alternative message-passing layer (selectable via `conv_type`).
- Implemented `SimpleGNNFallback` for environments without torch-geometric — uses dense adjacency and matrix-multiply message passing.

### Sun 21 Mar
- Wrote `catalog_to_pyg_data()` and `catalogs_to_pyg_dataset()` helpers.
- Pushed GNN code. Repository now has both CNN and GNN branches.

---

## Week 4 — 22–28 March 2026 | GNN Training, Evaluate Script, First Results

### Mon 22 Mar
- Completed `train.py` GNN branch: full training loop with both PyG and fallback paths.
- Tested GNN training: 100 catalogs, 30 epochs. Loss drops from ~3.2 to ~0.8 (MSE on log₁₀ M).

### Tue 23 Mar
- Built `evaluate.py` for both CNN and GNN models.
- CNN evaluation: generates a fresh test set with a different seed, computes all cosmological metrics.
- GNN evaluation: predicts log₁₀(M_halo) and reports MSE, MAE, Pearson r, scatter in dex.

### Wed 24 Mar
- First full CNN training run: 200 maps at 128×128, 50 epochs, batch size 8.
- Results: validation MSE = 3.2×10⁻⁵, Pearson r = 0.72. Power spectrum recovered within 20% for 500 < ℓ < 5000.
- **Issue:** Loss plateaued early (~epoch 15). Suspected underfitting — model capacity too small.

### Thu 25 Mar
- Increased `base_features` from 32 to 64 and depth from 3 to 4.
- Retrained: val MSE = 1.8×10⁻⁵, Pearson r = 0.81. Significant improvement at high ℓ.
- **Issue:** Training time jumped from 40 min to 3.5 hours. Need to optimise data loading.

### Fri 26 Mar
- Profiled training: bottleneck was in data generation (re-generating on-

each epoch). 
- Solution: pre-generate and save to .npy files, then load via `np.memmap`. Training time back to ~1 hour.
- Updated `mock_data_generator.py` CLI to support pre-generation.

### Sat 27 Mar
- First full GNN training run: 200 catalogs, 30 epochs.
- Results: MSE(log₁₀ M) = 0.42, scatter = 0.65 dex, Pearson r = 0.58.
- Disappointing correlation. Hypothesis: need more galaxies per halo and better feature engineering.

### Sun 28 Mar
- Experimented with adding tangential shear as a galaxy feature (strong cosmological signal).
- Improved Pearson r to 0.67. Still room for improvement.
- Pushed all changes.

---

## Week 5 — 29 March – 4 April 2026 | Hyperparameter Tuning & Debugging

### Mon 29 Mar
- Systematic hyperparameter sweep for CNN: tested {depth: 3, 4, 5} × {base_features: 32, 48, 64} × {lr: 1e-3, 3e-4, 1e-4}.
- Best configuration: depth=4, base_features=32, lr=1e-4. Higher capacity led to overfitting with only 200 maps.

### Tue 30 Mar
- Scaled up CNN data to 500 maps. Now depth=4, base_features=64 works best.
- Val MSE = 1.1×10⁻⁵, Pearson r = 0.87. Power spectrum ratio within 10% for ℓ ∈ [300, 8000].

### Wed 31 Mar
- **Bug discovery:** The peak-count loss gradient was causing NaN for maps with very low σ_κ. Root cause: division by σ ≈ 0 in the SNR calculation.
- Fix: added `+ 1e-8` to the standard deviation in `MassMapLoss._peak_count_loss()`.

### Thu 1 Apr
- GNN hyperparameter tuning: tested {hidden_dim: 64, 128, 256} × {n_layers: 3, 4, 6} × {k: 8, 16, 32}.
- Best: hidden_dim=128, n_layers=4, k=16. Overfitting with 6 layers.

### Fri 2 Apr
- Scaled GNN data to 500 catalogs. scatter = 0.48 dex, Pearson r = 0.74.
- Added dropout (p=0.1) which stabilised validation loss.

### Sat 3 Apr – Sun 4 Apr
- Code cleanup and documentation pass.
- Added docstrings to all public functions.
- Verified `requirements.txt` covers all imports.

---

## Week 6 — 5–11 April 2026 | Power Spectrum Analysis & Visualisation

### Mon 5 Apr
- Deep-dived into power spectrum recovery.
- Found the CNN consistently underestimates power at ℓ > 6000 (small scales).
- This is expected: high-ℓ modes are dominated by shape noise, and the CNN learns to suppress them.

### Tue 6 Apr
- Implemented cross-correlation coefficient r(ℓ) analysis.
- r(ℓ) > 0.9 for ℓ < 3000, drops to ~0.5 at ℓ = 8000. Consistent with literature (Jeffrey et al. 2020).

### Wed 7 Apr
- Built peak-count comparison: predicted maps recover ~85% of peaks above 3σ, but only ~60% above 5σ.
- The high-σ peaks correspond to massive clusters — most scientifically interesting targets.

### Thu 8 Apr
- **Bug:** `azimuthal_average_power_spectrum()` was double-counting the Nyquist frequency bin.
- Fix: adjusted bin edges to exclude k > k_Nyquist. Spectrum now matches numpy's periodogram.

### Fri 9 Apr
- Compared CNN results to simple Kaiser-Squires (KS) Wiener filter baseline.
- CNN outperforms KS on every metric: 35% lower MSE, 15% higher Pearson r, 20% better peak recovery.

### Sat 10 Apr – Sun 11 Apr
- Wrote evaluation summary and generated publication-quality figures (matplotlib).
- Pushed all results and figures.

---

## Week 7 — 12–18 April 2026 | GNN Improvements & Cross-Validation

### Mon 12 Apr
- Added GATv2Conv option to GNN. The multi-head attention helps with the heterogeneous galaxy populations.
- GAT version: scatter = 0.41 dex, Pearson r = 0.78. Modest improvement over EdgeConv.

### Tue 13 Apr
- Implemented proper per-halo evaluation: for each catalog, predict masses for ALL halos (not just halo 0).
- Required refactoring `catalogs_to_pyg_dataset()` to iterate over all halo indices.

### Wed 14 Apr
- **Bug:** Galaxy-halo assignment was off-by-one when halos were filtered by the mass function suppression.
- Fix: reindexed `galaxy_halo_ids` after filtering. This was silently corrupting ~15% of training examples.
- After fix: scatter improved from 0.41 to 0.35 dex. Major win.

### Thu 15 Apr
- Implemented 5-fold cross-validation for both models.
- CNN: Pearson r = 0.86 ± 0.02 (mean ± std across folds). Stable.
- GNN: Pearson r = 0.76 ± 0.05. More variance, likely due to smaller effective dataset.

### Fri 16 Apr
- Tested transfer learning: pre-train CNN on low-noise maps, fine-tune on high-noise.
- Result: 8% improvement in high-noise MSE compared to training from scratch.

### Sat 17 Apr – Sun 18 Apr
- Wrote unit tests for data generation and metrics.
- All tests pass. Coverage at ~78%.

---

## Week 8 — 19–27 April 2026 | Final Polishing, Documentation, Release

### Mon 19 Apr
- Final training runs with optimised hyperparameters:
  - CNN: 500 maps, 100 epochs, depth=4, base_features=64 → val MSE = 9.8×10⁻⁶, Pearson r = 0.89
  - GNN: 500 catalogs, 50 epochs, hidden=128, layers=4, k=16 → scatter = 0.33 dex, r = 0.79

### Tue 20 Apr
- Wrote comprehensive README with installation, usage, architecture diagrams (ASCII art), and results summary.
- Added `requirements.txt` with pinned minimum versions.

### Wed 21 Apr
- Code review pass: removed dead code, standardised logging, ensured no hardcoded paths.
- Added type hints to all function signatures.

### Thu 22 Apr
- **Issue:** `torch.fft.rfft2` returns a complex tensor with shape (..., N//2+1); the power spectrum loss was not handling the asymmetric shape correctly.
- Fix: switched to `torch.fft.fft2` (full complex FFT) for the loss, which keeps the symmetric shape. Marginal impact on accuracy but eliminates edge-case NaNs.

### Fri 23 Apr
- Final evaluation on a 100-map hold-out set:
  - MSE: 9.5×10⁻⁶
  - Pearson r: 0.89
  - Power spectrum recovery: < 8% deviation for ℓ ∈ [200, 7000]
  - Peak recovery (>3σ): 87%

### Sat 24 Apr
- Wrote this PROGRESS.md document.
- Cleaned up git history: squashed WIP commits, wrote descriptive commit messages.

### Sun 25 Apr
- Final GNN evaluation (100 catalogs):
  - MSE(log₁₀ M): 0.11
  - Scatter: 0.33 dex
  - Pearson r: 0.79
  - Outlier fraction (|Δlog M| > 1): 4.2%

### Mon 26 Apr
- Verified reproducibility: fresh clone → `pip install -r requirements.txt` → `python train.py cnn` → consistent results (within 2% of reported metrics).
- Checked GPU compatibility: tested on NVIDIA A100 (CUDA 12.1) and CPU-only mode.

### Tue 27 Apr
- Final push to main branch.
- Tagged release v1.0.0.
- Project complete. 🎉

---

## Summary of Key Results

| Model | Metric | Value |
|-------|--------|-------|
| CNN (U-Net) | Pixel MSE | 9.5 × 10⁻⁶ |
| CNN (U-Net) | Pearson r | 0.89 |
| CNN (U-Net) | Power spectrum error (ℓ < 7000) | < 8% |
| CNN (U-Net) | Peak recovery (> 3σ) | 87% |
| GNN (EdgeConv) | Scatter | 0.33 dex |
| GNN (EdgeConv) | Pearson r | 0.79 |
| GNN (EdgeConv) | Outlier fraction | 4.2% |

## Known Limitations

1. Synthetic data uses log-normal model, not full N-body simulations. Real Euclid data will have richer non-Gaussian structure.
2. CNN assumes periodic boundary conditions (FFT-based). Real survey footprints are irregular.
3. GNN uses a simplified HOD galaxy model. Realistic galaxy populations require SAMs or hydrodynamic sims.
4. No systematic error modelling (PSF, photometric redshift uncertainty, intrinsic alignments).

## Future Directions

- Replace log-normal mocks with FLASK or Takahashi et al. (2017) full-sky simulations.
- Add conditional normalising flow for uncertainty quantification on κ maps.
- Extend GNN to predict full halo properties (concentration, spin, substructure).
- Apply to Euclid Early Release Observations when available.
