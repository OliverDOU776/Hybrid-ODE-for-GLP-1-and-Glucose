import torch
import torch.nn as nn
from torchdiffeq import odeint

# Define the mechanistic ODE component (f_physio)
class MechanisticODE(nn.Module):
    def __init__(self, params):
        super(MechanisticODE, self).__init__()
        # params is a dictionary of fixed parameters
        self.params = params

    def forward(self, t, y):
        # y is a tensor of shape [batch_size, state_dim]
        # Unpack states: [Glucose, Insulin, Glucagon, FFA, GastricEmptying, GIP]
        G, I, Glu, FFA, E, GIP = torch.chunk(y, 6, dim=1)
        
        # Unpack parameters
        k1 = self.params['k1']
        k2 = self.params['k2']
        k3 = self.params['k3']
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        delta = self.params['delta']
        Glu_b = self.params['Glu_b']
        lambda_ = self.params['lambda']
        nu = self.params['nu']
        FFA_b = self.params['FFA_b']
        kappa = self.params['kappa']
        xi = self.params['xi']
        E_b = self.params['E_b']
        # Note: In the mechanistic part we do not model the meal input and GIP here.
        
        # Compute derivatives according to the ODEs (simplified version)
        dG_dt   = - (k1 * I + k2 * Glu) * G + k3 * E  # meal input added externally
        dI_dt   = alpha * G - beta * I
        dGlu_dt = - gamma * Glu + delta * (Glu_b - Glu)
        dFFA_dt = - lambda_ * FFA + nu * (FFA_b - FFA)
        dE_dt   = - kappa * E + xi * (E_b - E)
        dGIP_dt = torch.zeros_like(GIP)  # mechanistic part leaves GIP unchanged
        
        # Concatenate derivatives
        dydt = torch.cat([dG_dt, dI_dt, dGlu_dt, dFFA_dt, dE_dt, dGIP_dt], dim=1)
        return dydt

# Define the NN correction term (g_NN)
class NNCorrection(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=6):
        super(NNCorrection, self).__init__()
        # Here the input is the current state concatenated with time (t) -> dim = state_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, y):
        # Append time as an extra feature (broadcasted to batch dimension)
        t_feature = t.expand(y.size(0), 1)
        input_tensor = torch.cat([y, t_feature], dim=1)
        correction = self.net(input_tensor)
        return correction

# Define the full Hybrid ODE: dX/dt = f_physio(X;theta) + g_NN(X, t;phi)
class HybridODE(nn.Module):
    def __init__(self, mech_model, nn_correction):
        super(HybridODE, self).__init__()
        self.mech_model = mech_model
        self.nn_correction = nn_correction

    def forward(self, t, y):
        dydt_mech = self.mech_model(t, y)
        dydt_nn = self.nn_correction(t, y)
        return dydt_mech + dydt_nn

# Example usage: create a hybrid model with fixed parameters.
def create_hybrid_model():
    # Define parameters (use torch.tensor for compatibility)
    params = {
        'k1': torch.tensor(0.16),
        'k2': torch.tensor(0.04),
        'k3': torch.tensor(0.05),
        'alpha': torch.tensor(0.5),
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
    mech_model = MechanisticODE(params)
    nn_correction = NNCorrection(input_dim=7, hidden_dim=64, output_dim=6)
    hybrid_model = HybridODE(mech_model, nn_correction)
    return hybrid_model

if __name__ == '__main__':
    # Quick test: integrate the hybrid model from t=0 to 300 minutes.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_hybrid_model().to(device)
    t = torch.linspace(0, 300, 301).to(device)
    # Initial condition: [Glucose, Insulin, Glucagon, FFA, GastricEmptying, GIP]
    y0 = torch.tensor([[5.0, 10.0, 60.0, 0.5, 1.5, 0.0]], device=device)
    pred_y = odeint(model, y0, t)
    print("Hybrid ODE-NN integration successful!")
