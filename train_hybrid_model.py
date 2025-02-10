import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from hybrid_model import create_hybrid_model

# Assume synthetic data has been generated (or load from CSV)
def load_synthetic_data(csv_path='synthetic_metabolic_data.csv'):
    df = pd.read_csv(csv_path)
    # Convert to numpy array (exclude the Time column)
    data = df.drop(columns=['Time']).values
    # Normalize or convert as needed; here we assume the data are already scaled.
    return data

def ode_residual(model, t, y_pred):
    """
    Compute the residual of the ODE: || dy/dt - (f_physio + g_NN) ||^2
    y_pred is the integrated trajectory from the model.
    """
    # Compute time derivatives using finite differences
    dt = t[1] - t[0]
    dydt_numeric = (y_pred[1:] - y_pred[:-1]) / dt
    # Compute model predictions for derivatives at midpoints (simple approximation)
    t_mid = 0.5 * (t[:-1] + t[1:])
    y_mid = 0.5 * (y_pred[:-1] + y_pred[1:])
    dydt_model = torch.stack([model(ti, yi.unsqueeze(0)).squeeze(0) for ti, yi in zip(t_mid, y_mid)])
    residual = torch.mean((dydt_numeric - dydt_model) ** 2)
    return residual

def train_model(num_epochs=300, learning_rate=1e-3, lambda_ode=0.1, lambda_reg=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_hybrid_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load or generate synthetic data
    try:
        data_np = load_synthetic_data()
        print("Loaded synthetic data from CSV.")
    except Exception as e:
        print("Could not load CSV; generating dummy synthetic data.")
        # Generate dummy data using the mechanistic model only (for example purposes)
        data_np = np.random.rand(301, 6)
    
    # Convert synthetic data to torch tensor (batch_size=1)
    data_tensor = torch.tensor(data_np, dtype=torch.float32, device=device)
    t = torch.linspace(0, 300, data_tensor.size(0)).to(device)
    
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Integrate model from initial condition y0 (first row of data)
        y0 = data_tensor[0].unsqueeze(0)
        y_pred = odeint(model, y0, t)
        
        # Data fidelity loss: compare prediction to observed data
        loss_data = criterion(y_pred, data_tensor)
        # ODE residual loss: enforce that the derivative of y_pred matches the model
        loss_ode = ode_residual(model, t, y_pred)
        # Regularization loss (L2 on parameters)
        loss_reg = 0
        for param in model.parameters():
            loss_reg += torch.sum(param ** 2)
        loss_reg = lambda_reg * loss_reg
        
        loss = loss_data + lambda_ode * loss_ode + loss_reg
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Total Loss: {loss.item():.4f}, Data Loss: {loss_data.item():.4f}, ODE Loss: {loss_ode.item():.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "hybrid_model_trained.pth")
    print("Training complete and model saved.")
    
    # Plot the loss curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss")
    plt.show()
    
if __name__ == '__main__':
    train_model()
