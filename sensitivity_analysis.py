import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from hybrid_model import create_hybrid_model

def compute_rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def sensitivity_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define baseline synthetic data (for demonstration, use integration with the baseline model)
    model = create_hybrid_model().to(device)
    t = torch.linspace(0, 300, 301).to(device)
    y0 = torch.tensor([[5.0, 10.0, 60.0, 0.5, 1.5, 0.0]], device=device)
    with torch.no_grad():
        y_true = odeint(model, y0, t)
    
    # Define ranges for parameters to perturb
    param_names = ['k1', 'alpha']
    perturbations = np.linspace(-0.2, 0.2, 11)  # ±20% change
    rmse_results = {name: [] for name in param_names}
    
    for name in param_names:
        for perturb in perturbations:
            # Clone the model and modify the corresponding parameter in the mechanistic part
            model_perturbed = create_hybrid_model().to(device)
            # Access the mechanistic model parameters from the first sub-module
            param_tensor = getattr(model_perturbed.mech_model.params[name], 'clone')()
            new_value = param_tensor * (1.0 + perturb)
            model_perturbed.mech_model.params[name] = new_value.to(device)
            with torch.no_grad():
                y_pred = odeint(model_perturbed, y0, t)
            rmse = compute_rmse(y_pred, y_true)
            rmse_results[name].append(rmse)
    
    # Plot the sensitivity analysis results
    plt.figure(figsize=(8, 4))
    for name in param_names:
        plt.plot(perturbations * 100, rmse_results[name], label=name)
    plt.xlabel("Percentage Change in Parameter (%)")
    plt.ylabel("RMSE")
    plt.title("Sensitivity Analysis")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    sensitivity_analysis()
