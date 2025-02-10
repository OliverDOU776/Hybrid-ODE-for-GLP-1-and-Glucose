# Hybrid ODE-Neural Network Framework for Modeling GLP-1-Mediated Glucose Dynamics

This repository contains the complete code base used in our study, which integrates mechanistic ordinary differential equations (ODEs) with neural network (NN) corrections to model GLP-1-mediated glucose regulation. The repository provides scripts for synthetic data generation, model implementation, training, Bayesian parameter inference, sensitivity analysis, and uncertainty quantification.

## Overview

The repository is organized as follows:

- **Synthetic Data Generation**
  - **`generate_dataset.py`**  
    Generates a synthetic metabolic dataset simulating key variables (Glucose, Insulin, Glucagon, Free Fatty Acids (FFA), Gastric Emptying, and Gastric Inhibitory Polypeptide (GIP)) over a 5-hour period. A realistic meal input is modeled using a Gaussian function, and Gaussian noise is added to mimic measurement errors. The generated data are saved as a CSV file.

- **Hybrid ODE-NN Model Implementation**
  - **`hybrid_model.py`**  
    Implements the core hybrid ODE-NN model in PyTorch. This model combines a mechanistic ODE component (based on literature-derived parameters) with a neural network correction term. The model leverages the differentiable ODE solver provided by [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

- **Model Training**
  - **`train_hybrid_model.py`**  
    Trains the hybrid ODE-NN model using the synthetic dataset. The training script defines a multi-part loss function that includes data fidelity, ODE residual enforcement, and regularization terms. It then optimizes the model parameters using gradient-based methods and saves the trained model.

- **Bayesian Parameter Inference**
  - **`bayesian_inference.py`**  
    Uses Pyro to perform Bayesian inference on selected model parameters (e.g., \(k_1\) and \(\alpha\)). This script defines prior distributions, a likelihood function (assuming Gaussian noise), and employs the No-U-Turn Sampler (NUTS) to generate posterior samples. Posterior samples are printed and can be used for further analysis.

- **Sensitivity Analysis**
  - **`sensitivity_analysis.py`**  
    Performs sensitivity analysis by perturbing key parameters (e.g., \(k_1\) and \(\alpha\)) and computing the root mean square error (RMSE) between the perturbed and baseline predictions. The script plots the effect of parameter perturbations on model error to help identify the most influential parameters.

- **Uncertainty Quantification**
  - **`uncertainty_quantification.py`**  
    Demonstrates uncertainty quantification by using posterior samples (from Bayesian inference) to perform posterior predictive simulations. The script plots the mean prediction and the 95% credible intervals (uncertainty bounds) for glucose dynamics.

## Requirements

- **Python 3.x**
- Python libraries:
  - numpy
  - scipy
  - matplotlib
  - pandas
  - torch (PyTorch)
  - torchdiffeq
  - pyro-ppl (Pyro)

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib pandas torch torchdiffeq pyro-ppl
```

## Usage

### 1. Generate Synthetic Data
Run the synthetic data generator to create a dataset:

```bash
python generate_dataset.py
```

This will generate a CSV file named `synthetic_metabolic_data.csv` containing the simulated data.

### 2. Train the Hybrid Model
Train the hybrid ODE-NN model using the synthetic dataset:

```bash
python train_hybrid_model.py
```

The script will output training loss information, plot the loss curve, and save the trained model as `hybrid_model_trained.pth`.

### 3. Perform Bayesian Inference
Run the Bayesian inference script to estimate posterior distributions for selected model parameters:

```bash
python bayesian_inference.py
```

The script uses Pyro’s NUTS sampler and prints the posterior samples to the console.

### 4. Conduct Sensitivity Analysis
Evaluate the sensitivity of key model parameters by running:

```bash
python sensitivity_analysis.py
```

The script will plot RMSE changes as a function of parameter perturbations.

### 5. Uncertainty Quantification
Generate posterior predictive uncertainty plots:

```bash
python uncertainty_quantification.py
```

This will display the mean glucose predictions along with the 95% credible intervals.

## Customization

- **Model Parameters:** You can modify parameter values (e.g., in `generate_dataset.py` or within the model definitions in `hybrid_model.py`) to explore different physiological scenarios.
- **Network Architecture:** Adjust the neural network architecture (e.g., number of layers, neurons, activation functions) in the `NNCorrection` class in `hybrid_model.py`.
- **Simulation Settings:** Change simulation duration, noise levels, and initial conditions in the respective scripts to suit your experimental design.



## Contact

For questions or suggestions, please contact Zijia Wang at zijia.wang18@imperial.ac.uk
