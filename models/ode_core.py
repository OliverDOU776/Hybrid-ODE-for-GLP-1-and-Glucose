"""
Pure mechanistic ODE system for GLP-1-mediated glucose dynamics.

Implements the core ODE equations (1-6) from the paper:
"A Hybrid ODE-NN Framework for Modeling and Guiding GLP-1-Mediated Glucose Dynamics"
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class ODECore(nn.Module):
    """
    Mechanistic ODE model for glucose-insulin-GLP1 dynamics.
    
    Implements equations 1-6 from the paper, representing:
    - Glucose-stimulated insulin secretion with GLP-1 potentiation
    - GLP-1 inhibition of glucagon secretion
    - Glucose-triggered L-cell GLP-1 secretion
    - Gastric emptying suppression
    - Free fatty acid (FFA) kinetics
    
    State variables:
        G: Glucose concentration (mmol/L)
        I: Insulin concentration (pmol/L)
        Glu: Glucagon concentration (pmol/L)
        GLP1: GLP-1 concentration (pmol/L)
        GE: Gastric emptying rate
        FFA: Free fatty acid concentration (mmol/L)
    """
    
    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize ODE model with physiological parameters.
        
        Args:
            params: Dictionary of model parameters. If None, uses default values.
        """
        super().__init__()
        
        # Default parameters (typical physiological values)
        default_params = {
            # Insulin dynamics
            'a_GI': 0.0104,      # Glucose-insulin sensitivity (1/min)
            'k_I': 0.025,        # Insulin clearance rate (1/min)
            'rho': 0.003,        # GLP-1 potentiation factor
            'G_b': 5.0,          # Basal glucose (mmol/L)
            'I_b': 60.0,         # Basal insulin (pmol/L)
            
            # Glucagon dynamics
            'E_max': 0.1,        # Maximum GLP-1 suppression effect
            'EC_50': 50.0,       # GLP-1 concentration for half-max effect (pmol/L)
            'Glu_b': 80.0,       # Basal glucagon (pmol/L)
            
            # GLP-1 dynamics
            'V_max': 9.0,        # Maximum GLP-1 secretion rate (pmol/L/min)
            'K_m': 7.0,          # Michaelis constant for glucose (mmol/L)
            'k_L': 0.02,         # GLP-1 degradation rate (1/min)
            
            # Gastric emptying
            'k_GE0': 0.01,       # Basal gastric emptying rate (1/min)
            'IGD_50': 1000.0,    # GD for half-max suppression
            'g': 2.0,            # Hill coefficient for GE suppression
            
            # FFA dynamics
            'p_7': 0.05,         # FFA clearance rate (1/min)
            'p_8': 0.001,        # Insulin suppression factor
            'p_9': 0.01,         # Glucose-driven lipolysis factor
        }
        
        # Update with provided parameters
        if params is not None:
            default_params.update(params)
        
        # Register parameters as buffers (non-trainable by default)
        for name, value in default_params.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
            
    def forward(self, t: torch.Tensor, state: torch.Tensor, 
                external_inputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute ODE derivatives for the current state.
        
        Args:
            t: Current time (scalar or batch)
            state: Current state vector [G, I, Glu, GLP1, GE, FFA] shape: (batch, 6) or (6,)
            external_inputs: Dictionary with optional inputs:
                - 'meal': Meal glucose input (mmol/L/min)
                - 'tVNS': Vagal nerve stimulation (binary 0/1)
                - 'GD': Gastric distension signal
                
        Returns:
            derivatives: State derivatives [dG/dt, dI/dt, dGlu/dt, dGLP1/dt, dGE/dt, dFFA/dt]
        """
        # Handle batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Extract state variables
        G = state[:, 0]      # Glucose
        I = state[:, 1]      # Insulin
        Glu = state[:, 2]    # Glucagon
        GLP1 = state[:, 3]   # GLP-1
        GE = state[:, 4]     # Gastric emptying (not used in basic equations)
        FFA = state[:, 5]    # Free fatty acids
        
        # Parse external inputs
        if external_inputs is None:
            external_inputs = {}
        meal = external_inputs.get('meal', torch.zeros_like(G))
        tVNS = external_inputs.get('tVNS', torch.zeros_like(G))
        GD = external_inputs.get('GD', torch.zeros_like(G))
        
        # Initialize derivatives
        derivatives = torch.zeros_like(state)
        
        # Equation 1/2: Insulin dynamics with GLP-1 potentiation
        # dI/dt = Pi(GLP1) * a_GI * (G - G_b) - k_I * (I - I_b)
        Pi = 1.0 + self.rho * GLP1  # GLP-1 potentiation factor
        dI_dt = Pi * self.a_GI * (G - self.G_b) - self.k_I * (I - self.I_b)
        
        # Equation 3: Glucagon dynamics with GLP-1 suppression
        # dGlu/dt = -E_max * (GLP1 / (EC_50 + GLP1)) * (Glu - Glu_b)
        glp1_effect = self.E_max * (GLP1 / (self.EC_50 + GLP1))
        dGlu_dt = -glp1_effect * (Glu - self.Glu_b)
        
        # Equation 4: GLP-1 dynamics
        # dGLP1/dt = V_max * (G / (K_m + G)) - k_L * GLP1
        glucose_stimulation = self.V_max * (G / (self.K_m + G))
        dGLP1_dt = glucose_stimulation - self.k_L * GLP1
        
        # Equation 5: Gastric emptying modulation (compute k_GE)
        # k_GE(t) = k_GE0 * [1 - (GD^g / (IGD_50^g + GD^g))]
        GD_effect = torch.pow(GD, self.g) / (torch.pow(self.IGD_50, self.g) + torch.pow(GD, self.g))
        k_GE = self.k_GE0 * (1.0 - GD_effect)
        
        # Equation 6: FFA dynamics
        # dFFA/dt = -p_7 * FFA - p_8 * I * FFA + p_9 * G * FFA
        dFFA_dt = -self.p_7 * FFA - self.p_8 * I * FFA + self.p_9 * G * FFA
        
        # Glucose dynamics (simplified - needs hepatic glucose production and utilization)
        # For now, using a simplified version with meal input and insulin-dependent uptake
        insulin_effect = 0.01 * (I - self.I_b)  # Insulin-dependent glucose uptake
        glucagon_effect = 0.005 * (Glu - self.Glu_b)  # Glucagon-stimulated production
        dG_dt = meal - insulin_effect + glucagon_effect - k_GE * G
        
        # Gastric emptying rate dynamics (placeholder)
        dGE_dt = torch.zeros_like(GE)
        
        # Assemble derivatives
        derivatives[:, 0] = dG_dt
        derivatives[:, 1] = dI_dt
        derivatives[:, 2] = dGlu_dt
        derivatives[:, 3] = dGLP1_dt
        derivatives[:, 4] = dGE_dt
        derivatives[:, 5] = dFFA_dt
        
        if squeeze_output:
            derivatives = derivatives.squeeze(0)
            
        return derivatives
    
    def get_steady_state(self, external_inputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute approximate steady-state values for the ODE system.
        
        Args:
            external_inputs: External inputs (should be zero for steady state)
            
        Returns:
            steady_state: Steady state vector [G_ss, I_ss, Glu_ss, GLP1_ss, GE_ss, FFA_ss]
        """
        # At steady state with no external inputs, use basal values
        steady_state = torch.zeros(6)
        steady_state[0] = self.G_b      # Glucose
        steady_state[1] = self.I_b      # Insulin  
        steady_state[2] = self.Glu_b    # Glucagon
        steady_state[3] = 0.0           # GLP1 (assuming minimal at steady state)
        steady_state[4] = 0.0           # GE
        steady_state[5] = 1.0           # FFA (normalized)
        
        return steady_state
    
    def check_mass_balance(self, state: torch.Tensor, derivatives: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Check mass balance constraints for validation.
        
        Args:
            state: Current state vector
            derivatives: Computed derivatives
            
        Returns:
            balances: Dictionary of mass balance checks
        """
        balances = {}
        
        # Ensure non-negative concentrations
        balances['non_negative'] = (state >= 0).all()
        
        # Check reasonable physiological ranges
        G = state[..., 0]
        I = state[..., 1]
        balances['glucose_range'] = (G >= 2.0) & (G <= 30.0)  # mmol/L
        balances['insulin_range'] = (I >= 0.0) & (I <= 1000.0)  # pmol/L
        
        return balances