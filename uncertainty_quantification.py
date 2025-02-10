
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from hybrid_model import MechanisticODE
import pyro

def posterior_predictive(posterior_samples, y0, t):
    # For each posterior sample, simulate the mechanistic model
    predictions = []
    for i in range(posterior_samples['k1'].size(0)):
        sample_params = {
            'k1': posterior_samples['k1'][i],
            'k2': torch.tensor(0.04),
            'k3': torch.tensor(0.05),
            'alpha': posterior_samples['alpha'][i],
            'beta': torch.tensor(0.1),
            'gamma': torch.tensor(0.8),
            'delta': torch.tensor(1.2),
            'Glu_b': torch.tensor(60.0),
            'lambda': torch.tensor(0.03),
            'nu': torch.tensor(0.15),
            'FFA_b': torch.tensor(0.5),
            'kappa': torch.tensor(0.04),
            'xi': torch.tensor(1.0),
            'E_b': torch.tensor(1.5)
        }
        mech_model = MechanisticODE(sample_params)
        y_sample = odeint(mech_model, y0, t)
        predictions.append(y_sample.cpu().numpy())
    return np.array(predictions)

def plot_uncertainty(predictions, t):
    # Compute mean and 95% credible intervals
    mean_pred = np.mean(predictions, axis=0)
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    
    # Plot for one variable (e.g., Glucose, index 0)
    plt.figure(figsize=(8, 4))
    plt.plot(t.cpu().numpy(), mean_pred[:, 0], 'b-', label='Mean Prediction (Glucose)')
    plt.fill_between(t.cpu().numpy(), lower[:, 0], upper[:, 0], color='b', alpha=0.3, label='95% Credible Interval')
    plt.xlabel("Time (min)")
    plt.ylabel("Glucose (mmol/L)")
    plt.title("Posterior Predictive Uncertainty")
    plt.legend()
    plt.show()

def run_uncertainty_quantification():
    # For demonstration, load posterior samples from bayesian_inference.py (here we simulate dummy samples)
    num_samples = 100
    posterior_samples = {
        'k1': torch.normal(0.16, 0.05, size=(num_samples,)),
        'alpha': torch.normal(0.5, 0.1, size=(num_samples,))
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(0, 300, 301).to(device)
    y0 = torch.tensor([[5.0, 10.0, 60.0, 0.5, 1.5, 0.0]], device=device)
    predictions = posterior_predictive(posterior_samples, y0, t)
    plot_uncertainty(predictions, t)

if __name__ == '__main__':
    run_uncertainty_quantification()
