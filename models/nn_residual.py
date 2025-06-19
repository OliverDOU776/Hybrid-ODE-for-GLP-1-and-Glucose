"""
Neural network residual block for the hybrid ODE-NN model.

Implements a 4-layer feedforward network that learns residual dynamics
to augment the mechanistic ODE model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class NNResidual(nn.Module):
    """
    Feed-forward neural network for learning residual dynamics.
    
    Architecture:
        - Input: [t, state (6), GLP1, tVNS] -> 9 features
        - Hidden: 4 layers × 64 units with ReLU activation
        - Output: 6 residual terms (one per state variable)
        
    The network is initialized to output zeros so the initial model
    matches the pure ODE solution.
    """
    
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, 
                 output_dim: int = 6, n_layers: int = 4,
                 activation: str = 'relu', dropout: float = 0.0):
        """
        Initialize the residual neural network.
        
        Args:
            input_dim: Input dimension (default 9: t + 6 states + GLP1 + tVNS)
            hidden_dim: Hidden layer dimension (default 64)
            output_dim: Output dimension (default 6: one per state)
            n_layers: Number of hidden layers (default 4)
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout probability (default 0.0)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Select activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize to output zeros
        self._initialize_zero_output()
        
    def _initialize_zero_output(self):
        """
        Initialize network weights to output zeros initially.
        
        This ensures the hybrid model starts with pure ODE dynamics.
        """
        # Zero-initialize the output layer
        with torch.no_grad():
            self.network[-1].weight.data.zero_()
            self.network[-1].bias.data.zero_()
            
        # Small random initialization for other layers
        for layer in self.network[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, t: torch.Tensor, state: torch.Tensor, 
                glp1: torch.Tensor, tvns: torch.Tensor) -> torch.Tensor:
        """
        Compute neural network residuals.
        
        Args:
            t: Time tensor (batch_size,) or scalar
            state: State vector (batch_size, 6) or (6,)
            glp1: GLP-1 concentration (batch_size,) or scalar
            tvns: Vagal nerve stimulation signal (batch_size,) or scalar
            
        Returns:
            residuals: NN residual terms (batch_size, 6) or (6,)
        """
        # Handle different input dimensions
        squeeze_output = False
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
            
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(state.shape[0])
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(state.shape[0])
            
        if glp1.dim() == 0:
            glp1 = glp1.unsqueeze(0).expand(state.shape[0])
        elif glp1.dim() == 1 and glp1.shape[0] == 1:
            glp1 = glp1.expand(state.shape[0])
            
        if tvns.dim() == 0:
            tvns = tvns.unsqueeze(0).expand(state.shape[0])
        elif tvns.dim() == 1 and tvns.shape[0] == 1:
            tvns = tvns.expand(state.shape[0])
        
        # Concatenate inputs
        # Features: [t, G, I, Glu, GLP1, GE, FFA, GLP1_external, tVNS]
        nn_input = torch.cat([
            t.unsqueeze(-1),      # Time
            state,                # All 6 state variables
            glp1.unsqueeze(-1),   # External GLP1 (could be different from state)
            tvns.unsqueeze(-1)    # Vagal stimulation
        ], dim=-1)
        
        # Forward pass through network
        residuals = self.network(nn_input)
        
        if squeeze_output:
            residuals = residuals.squeeze(0)
            
        return residuals
    
    def get_feature_importance(self, t: torch.Tensor, state: torch.Tensor,
                              glp1: torch.Tensor, tvns: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using gradient-based sensitivity.
        
        Args:
            t, state, glp1, tvns: Input tensors
            
        Returns:
            importance: Feature importance scores (input_dim,)
        """
        # Ensure we have batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if glp1.dim() == 0:
            glp1 = glp1.unsqueeze(0)
        if tvns.dim() == 0:
            tvns = tvns.unsqueeze(0)
        
        # Build input tensor
        nn_input = torch.cat([
            t.unsqueeze(-1),
            state,
            glp1.unsqueeze(-1),
            tvns.unsqueeze(-1)
        ], dim=-1)
        nn_input.requires_grad = True
        
        # Forward pass
        output = self.network(nn_input)
        
        # Compute gradients for each output
        importance = torch.zeros(self.input_dim)
        
        for i in range(self.output_dim):
            if nn_input.grad is not None:
                nn_input.grad.zero_()
                
            output[:, i].sum().backward(retain_graph=True)
            importance += nn_input.grad.abs().mean(dim=0)
        
        return importance / self.output_dim
    
    def regularization_loss(self, l2_weight: float = 1e-4, 
                           sparsity_weight: float = 0.0) -> torch.Tensor:
        """
        Compute regularization loss for the neural network.
        
        Args:
            l2_weight: L2 regularization weight
            sparsity_weight: Sparsity regularization weight
            
        Returns:
            reg_loss: Total regularization loss
        """
        reg_loss = 0.0
        
        # L2 regularization on all weights
        if l2_weight > 0:
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    reg_loss += l2_weight * layer.weight.pow(2).sum()
        
        # Sparsity regularization on activations (if needed)
        if sparsity_weight > 0:
            # This would require caching activations during forward pass
            pass
        
        return reg_loss