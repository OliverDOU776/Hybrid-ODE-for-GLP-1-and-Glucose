#!/usr/bin/env python3
"""
This script simulates a metabolic system using an ODE model that incorporates key variables 
involved in GLP-1-mediated glucose regulation, including Glucose, Insulin, Glucagon, FFA, 
Gastric Emptying, and GIP. A realistic meal input function and measurement noise are also added 
to produce a synthetic dataset, which is saved as a CSV file.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# Define a realistic meal input function using a Gaussian curve
def meal_input(t, peak_time=30, amplitude=20, width=10):
    """
    Gaussian meal input function.
    
    Parameters:
        t (float): Time (in minutes).
        peak_time (float): Time when meal input peaks.
        amplitude (float): Peak amplitude of the meal input.
        width (float): Standard deviation of the Gaussian curve.
    
    Returns:
        float: Meal input value at time t.
    """
    return amplitude * np.exp(-0.5 * ((t - peak_time) / width)**2)

# Define the ODE system representing the metabolic model
def metabolic_odes(y, t, params):
    """
    Defines the ODE system for metabolic dynamics.
    
    State vector y = [G, I, Glu, FFA, E, GIP] where:
      - G: Glucose concentration (mmol/L)
      - I: Insulin concentration (µU/mL)
      - Glu: Glucagon concentration (pg/mL)
      - FFA: Free Fatty Acids (mmol/L)
      - E: Gastric emptying rate (mL/min)
      - GIP: Gastric Inhibitory Polypeptide (pg/mL)
    
    Parameters are provided in the dictionary 'params'.
    """
    # Unpack state variables
    G, I, Glu, FFA, E, GIP = y
    
    # Unpack parameters
    k1 = params['k1']         # Insulin-mediated glucose uptake rate (min^-1)
    k2 = params['k2']         # Glucagon-stimulated glucose production rate (min^-1)
    k3 = params['k3']         # Gastric emptying contribution (min^-1)
    alpha = params['alpha']   # Insulin secretion rate (min^-1)
    beta = params['beta']     # Insulin degradation rate (min^-1)
    gamma = params['gamma']   # GLP-1 inhibitory effect on glucagon (min^-1)
    delta = params['delta']   # Basal glucagon secretion rate constant (min^-1)
    Glu_b = params['Glu_b']   # Basal glucagon level (pg/mL)
    lambda_ = params['lambda']  # Effect on FFA metabolism (min^-1)
    nu = params['nu']         # Basal FFA production rate (min^-1)
    FFA_b = params['FFA_b']   # Basal FFA level (mmol/L)
    kappa = params['kappa']   # Effect on gastric emptying (min^-1)
    xi = params['xi']         # Basal gastric emptying rate constant (min^-1)
    E_b = params['E_b']       # Basal gastric emptying rate (mL/min)
    a = params['a']           # Meal-induced GIP secretion coefficient
    b = params['b']           # GIP degradation rate
    
    # Meal input (simulated as a Gaussian function)
    D_t = meal_input(t, peak_time=params['meal_peak'], amplitude=params['meal_amplitude'], width=params['meal_width'])
    
    # ODEs for the system
    dG_dt   = - (k1 * I + k2 * Glu) * G + k3 * E + D_t
    dI_dt   = alpha * G - beta * I
    dGlu_dt = - gamma * Glu + delta * (Glu_b - Glu)
    dFFA_dt = - lambda_ * FFA + nu * (FFA_b - FFA)
    dE_dt   = - kappa * E + xi * (E_b - E)
    dGIP_dt = a * D_t - b * GIP
    
    return [dG_dt, dI_dt, dGlu_dt, dFFA_dt, dE_dt, dGIP_dt]

def simulate_metabolic_model(sim_time, y0, params, noise_std=0.05):
    """
    Simulates the metabolic model over the specified time and adds Gaussian noise to mimic measurement error.
    
    Parameters:
        sim_time (array): Array of time points (in minutes).
        y0 (list): Initial conditions for the state variables.
        params (dict): Dictionary of model parameters.
        noise_std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        tuple: (time vector, noisy solution, noise-free solution)
    """
    # Solve the ODE system (noise-free)
    true_solution = odeint(metabolic_odes, y0, sim_time, args=(params,))
    
    # Add Gaussian noise to simulate measurement errors
    noise = np.random.normal(0, noise_std, true_solution.shape)
    noisy_solution = true_solution + noise
    
    return sim_time, noisy_solution, true_solution

def main():
    # Define simulation time (0 to 300 minutes with 1-minute resolution)
    sim_time = np.linspace(0, 300, 301)
    
    # Initial conditions: [Glucose, Insulin, Glucagon, FFA, Gastric Emptying, GIP]
    y0 = [5.0,    # Glucose (mmol/L)
          10.0,   # Insulin (µU/mL)
          60.0,   # Glucagon (pg/mL)
          0.5,    # FFA (mmol/L)
          1.5,    # Gastric Emptying (mL/min)
          0.0]    # GIP (pg/mL) - starting at zero
    
    # Define model parameters based on literature values and assumptions
    params = {
        'k1': 0.16,         # Insulin-mediated glucose uptake rate (min^-1)
        'k2': 0.04,         # Glucagon-stimulated glucose production rate (min^-1)
        'k3': 0.05,         # Gastric emptying contribution (min^-1)
        'alpha': 0.5,       # Insulin secretion rate (min^-1)
        'beta': 0.1,        # Insulin degradation rate (min^-1)
        'gamma': 0.8,       # Inhibitory effect on glucagon (min^-1)
        'delta': 1.2,       # Basal glucagon secretion rate constant (min^-1)
        'Glu_b': 60.0,      # Basal glucagon level (pg/mL)
        'lambda': 0.03,     # Effect on FFA metabolism (min^-1)
        'nu': 0.15,         # Basal FFA production rate (min^-1)
        'FFA_b': 0.5,       # Basal FFA level (mmol/L)
        'kappa': 0.04,      # Effect on gastric emptying (min^-1)
        'xi': 1.0,          # Basal gastric emptying rate constant (min^-1)
        'E_b': 1.5,         # Basal gastric emptying rate (mL/min)
        'a': 0.3,           # GIP secretion coefficient
        'b': 0.1,           # GIP degradation rate
        'meal_peak': 30,    # Peak time of meal input (minutes)
        'meal_amplitude': 20, # Amplitude of meal input (arbitrary units)
        'meal_width': 10    # Width (std deviation) of meal input (minutes)
    }
    
    # Simulate the model with added noise
    t, noisy_solution, true_solution = simulate_metabolic_model(sim_time, y0, params, noise_std=0.05)
    
    # Create a pandas DataFrame for the noisy (observed) data
    df_noisy = pd.DataFrame(noisy_solution, columns=['Glucose', 'Insulin', 'Glucagon', 'FFA', 'GastricEmptying', 'GIP'])
    df_noisy['Time'] = t
    df_noisy = df_noisy[['Time', 'Glucose', 'Insulin', 'Glucagon', 'FFA', 'GastricEmptying', 'GIP']]
    
    # Save the dataset to a CSV file
    csv_filename = 'synthetic_metabolic_data.csv'
    df_noisy.to_csv(csv_filename, index=False)
    print(f"Synthetic dataset saved to '{csv_filename}'.")
    
    # Plot the results (both noise-free and noisy trajectories)
    plt.figure(figsize=(12, 10))
    
    variable_names = ['Glucose', 'Insulin', 'Glucagon', 'FFA', 'Gastric Emptying', 'GIP']
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(t, true_solution[:, i], 'b-', label='True')
        plt.plot(t, noisy_solution[:, i], 'r.', alpha=0.5, label='Noisy')
        plt.title(variable_names[i])
        plt.xlabel('Time (min)')
        plt.ylabel(variable_names[i])
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
