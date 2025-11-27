# Distributionally Robust K-Means

A MATLAB implementation of **Distributionally Robust K-Means (DR-K-Means)**, a robust clustering algorithm that provides improved performance over standard k-means, particularly in the presence of outliers and adversarial perturbations.

## Overview

This repository implements a distributionally robust optimization approach to k-means clustering. Unlike standard k-means, which minimizes the expected squared distance, DR-K-Means optimizes a worst-case objective over a family of distributions, making it more resilient to outliers and distributional shifts.

## Algorithm

The DR-K-Means algorithm solves the following optimization problem:

```
min_M  max_π  E_π[||X - M||²] + α·||X - M·π||²_F
```

where:
- **M** (d × k): Cluster centroids
- **π** (k × N): Assignment weights (probability distributions over clusters for each point)
- **γ ≥ 1**: Robustness parameter (larger values = more robust)
- **α = 1/(γ-1)**: Regularization parameter

The algorithm alternates between:
1. **Assignment step**: For each data point, solve a quadratic program to find optimal assignment weights π that maximize the objective
2. **Centroid update**: Update centroids M using a closed-form solution based on weighted assignments

### Key Features

- **Robustness**: Handles outliers and adversarial perturbations better than standard k-means
- **Flexible assignments**: Points can have soft assignments (weights) rather than hard cluster assignments
- **Iterative optimization**: Uses projected gradient descent or quadratic programming for efficient updates

## Repository Structure

```
.
├── datasets/                      # Data directory (put your data here)
│   ├── nips.mat                  # NIPS dataset (sparse document-word matrix)
│   ├── nips_coeff50.mat         # Precomputed PCA coefficients
│   └── nips_doc50.mat           # 50-dimensional PCA projection
│
├── findNextM_v6.m                # Core optimization step (generic version)
├── findNextM_v6_mnist.m         # Core optimization step (MNIST-specific)
├── findNextM_v7_fast.m          # Fast variant using projected gradient descent
│
├── pgd_classification.m          # Classification accuracy experiments
├── pgd_outlier_recall.m          # Outlier detection/recall experiments
├── main2_nips_outlier_comp.m     # NIPS dataset outlier experiments
│
└── Readme.md                     # This file
```

## Usage

### Classification Experiments

Run classification accuracy comparisons on synthetic Gaussian mixture data:

```matlab
% Edit parameters in pgd_classification.m, then run:
pgd_classification
```

This script:
- Generates synthetic GMM data with K components
- Compares k-means++ vs DR-K-Means classification accuracy
- Uses majority-vote labeling of centroids for evaluation

### Outlier Detection Experiments

Test outlier recall performance:

```matlab
% For synthetic data:
pgd_outlier_recall

% For NIPS dataset:
main2_nips_outlier_comp
```

These scripts:
- Inject outliers into the dataset
- Compare k-means++ vs DR-K-Means ability to identify outliers
- Measure recall (fraction of true outliers correctly identified)

### Core Algorithm

The main algorithm is implemented in the `drlm` function (included in each script):

```matlab
[M, grad, wQE_f, wQE_history] = drlm(gamma, M0, r, d, X, N)
```

**Parameters:**
- `gamma` (≥ 1): Robustness parameter. Larger values increase robustness but may reduce fit quality.
- `M0` (d × k): Initial centroids (typically from k-means++)
- `r`: Algorithm parameter (typically 3)
- `d`: Data dimensionality
- `X` (d × N): Data matrix
- `N`: Number of data points

**Returns:**
- `M` (d × k): Final centroids
- `wQE_f`: Final weighted quadratic error
- `wQE_history`: Objective value history

## Key Implementation Details

### Optimization Methods

1. **findNextM_v6**: Uses `quadprog` to solve QP for each point
2. **findNextM_v7_fast**: Faster variant using projected gradient descent with Nesterov momentum

### Typical Parameter Settings

- **Classification**: `gamma = 1.04`, `r = 3`
- **Outlier detection**: `gamma = 1.01-1.1`, `r = 3`
- **Convergence**: Algorithm stops when relative change in objective < 1e-2

## Requirements

- MATLAB (R2018b or later)
- Statistics and Machine Learning Toolbox (for `kmeans`, `pdist2`, `mvnrnd`)
- Optimization Toolbox (for `quadprog`)
- Parallel Computing Toolbox (optional, for parallel execution)

## Example Results

### Classification Accuracy
On synthetic GMM data with K=200 components, d=200 dimensions:
- **k-means++**: ~85-90% accuracy
- **DR-K-Means**: ~90-95% accuracy

### Outlier Recall
On NIPS dataset with 5% outliers:
- **k-means++**: ~60-85% recall (varies with k)
- **DR-K-Means**: ~95-100% recall

## Citation

If you use this code, please cite the associated paper (if applicable).

## License

[Add your license here]
