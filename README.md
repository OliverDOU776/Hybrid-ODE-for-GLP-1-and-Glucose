# Hybrid ODE-NN Framework for GLP-1-Mediated Glucose Dynamics

This repository contains the implementation of the hybrid ODE-neural network framework from the paper *"A Hybrid ODE–NN Framework for Modeling and Guiding GLP-1–Mediated Glucose Dynamics"*.

## Overview

The framework combines mechanistic ordinary differential equations (ODEs) with neural networks to model glucose-insulin-GLP1 dynamics in response to meals and therapeutic interventions. The model:

- Integrates physiological knowledge through mechanistic ODEs (equations 1-6)
- Learns residual dynamics using a 4-layer neural network
- Provides uncertainty quantification through variational inference
- Enforces physics constraints via a multi-component loss function

## Key Features

- **Hybrid Architecture**: Combines interpretable ODEs with flexible neural networks
- **Bayesian Inference**: Uncertainty quantification via variational inference or MCMC
- **Physics-Informed**: Incorporates ODE constraints in the loss function
- **Multiple Solvers**: Supports various ODE solvers (dopri5, rk45, radau, etc.)
- **Clinical Ready**: Handles both synthetic and real clinical data
- **Well-Tested**: Comprehensive test suite covering ODE gradients, model training, and more

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)
- ~1GB disk space for data and checkpoints

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Hybrid-ODE-for-GLP-1-and-Glucose.git
cd Hybrid-ODE-for-GLP-1-and-Glucose

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (optional, for GPU)
# See https://pytorch.org for platform-specific instructions
```

## Quick Start

```bash
# Run tests to verify installation
pytest tests/ -v

# Train model with default settings (300 epochs)
python train/train_hybrid.py --data data/4gi_dataset.csv

# Quick training for testing (edit configs/default.yaml to reduce epochs)
python train/train_hybrid.py --data data/4gi_dataset.csv

# Generate all figures from the paper
python plots/plot_all.py --checkpoint checkpoints/experiment_name/best_model.pth --figures all

# Generate specific figures (faster)
python plots/plot_all.py --checkpoint checkpoints/experiment_name/best_model.pth --figures fig1 fig3 tables
```

## Model Architecture

### Mechanistic ODE Core

The model implements 6 coupled ODEs describing:

1. **Insulin dynamics** with GLP-1 potentiation
2. **Glucagon suppression** by GLP-1  
3. **GLP-1 secretion** triggered by glucose
4. **Gastric emptying** modulation
5. **Free fatty acid** kinetics

### Neural Network Residual

- 4-layer feedforward network (64 hidden units, ReLU activation)
- Zero-initialized to preserve ODE solution initially
- Learns corrections to mechanistic model

### Loss Function

```
L = L_data + λ₁·L_physics + λ₂·L_Bayesian
```

Where:
- `L_data`: Mean squared error between predictions and observations
- `L_physics`: ODE residual matching constraint
- `L_Bayesian`: KL divergence for variational inference

## Usage Examples

### Training a Model

```bash
# Basic training
python train/train_hybrid.py --data data/4gi_dataset.csv

# With variational inference
python train/train_hybrid.py --data data/4gi_dataset.csv --vi

# Using configuration file
python train/train_hybrid.py --config configs/4gi_vi.yaml
```

### Ablation Studies

```bash
# Pure ODE model (no neural network)
python train/train_hybrid.py --no-nn

# No physics constraints
python train/train_hybrid.py --no-physics  

# No Bayesian inference
python train/train_hybrid.py --no-bayes
```

### Evaluation

```bash
# Evaluate trained model
python eval/evaluate.py checkpoints/experiment/best_model.pth --data data/4gi_dataset.csv

# Generate figures from paper
python plots/plot_all.py --checkpoint checkpoints/experiment/best_model.pth --figures all
```

## Project Structure

```
├── models/                 # Model implementations
│   ├── ode_core.py        # Mechanistic ODE system
│   ├── nn_residual.py     # Neural network component
│   ├── hybrid_ode_nn.py   # Combined hybrid model
│   └── bayes.py           # Bayesian utilities
├── inference/             # Inference methods
│   ├── vi.py             # Variational inference
│   └── mcmc.py           # MCMC sampling
├── train/                 # Training scripts
│   └── train_hybrid.py   # Main training script
├── eval/                  # Evaluation tools
│   └── evaluate.py       # Metrics computation
├── plots/                 # Visualization
│   └── plot_all.py       # Generate paper figures
├── data/                  # Data files and loaders
│   ├── 4gi_dataset.csv   # Synthetic 4GI data
│   ├── generate4GI.py    # 4GI data generator
│   └── download_mimic.py # MIMIC data downloader
├── configs/              # Configuration files
│   ├── default.yaml     # Default settings
│   ├── 4gi_*.yaml       # 4GI experiments
│   └── mimic_*.yaml     # Clinical experiments
├── tests/                # Unit tests
│   ├── test_ode_jacobians.py    # ODE gradient tests
│   ├── test_gradient_correctness.py  # Model gradient flow
│   └── test_training.py         # End-to-end training
└── requirements.txt     # Python dependencies
```

## Configuration

Key parameters can be set via YAML configs:

```yaml
model:
  nn_hidden: 64        # NN hidden units
  nn_layers: 4         # NN depth
  solver: "dopri5"     # ODE solver

training:
  epochs: 300
  batch_size: 32
  learning_rate: 1e-3
  lambda1: 1.0         # Physics loss weight
  lambda2: 1.0         # Bayesian loss weight
```

## Results

The model achieves:
- **RMSE < 0.5 mmol/L** for glucose predictions
- **95% coverage** of true values within uncertainty bands
- **Interpretable parameters** with physiological meaning

## Testing

Run the test suite to verify correct installation:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_ode_jacobians.py -v
pytest tests/test_gradient_correctness.py -v
pytest tests/test_training.py -v

# Run with coverage report
pytest tests/ --cov=models --cov=train --cov=inference
```

## Citation

If you use this code, please cite:

```bibtex
@article{hybrid_ode_nn_glp1,
  title={A Hybrid ODE-NN Framework for Modeling and Guiding GLP-1-Mediated Glucose Dynamics},
  author={[Zijia Wang, Sumbal Sarwar, Christofer Toumazou]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Figures and Visualization

The framework can generate three main figures from the paper:

- **Figure 1**: Model predictions vs ground truth for glucose, insulin, and GLP-1
- **Figure 2**: Sensitivity analysis showing parameter importance (computationally intensive)
- **Figure 3**: Predictions with uncertainty bands (requires VI/MCMC)

See `python plots/plot_all.py --help` for all options.

## Implementation Notes

- The ODE solver uses SciPy's `solve_ivp` with automatic mapping from torchdiffeq solver names
- Neural network residuals are zero-initialized to preserve mechanistic ODE behavior initially
- Variational parameters use underscores instead of dots for PyTorch compatibility
- Physics loss is computed using finite differences for ODE constraint enforcement
- Variational inference (--vi flag) is currently under development; use MCMC for uncertainty quantification

## Known Issues

- Variational inference training currently falls back to standard training (device placement issue)
- MCMC sampling after training may encounter device errors but saves MAP estimates correctly
- Sensitivity analysis (Figure 2) requires ~5-10 minutes due to many model evaluations

## Acknowledgments

- PhysioNet for MIMIC-IV data access
- PyTorch team for deep learning framework
- SciPy team for ODE solvers