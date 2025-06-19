"""
Hybrid ODE-NN model combining mechanistic ODEs with neural network residuals.

This is the main model class that integrates the ODE core with NN augmentation
and provides training/inference interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
from .ode_core import ODECore
from .nn_residual import NNResidual
from .bayes import VariationalParameters, bayes_loss
import logging

logger = logging.getLogger(__name__)


class HybridODENN(nn.Module):
    """
    Hybrid model combining mechanistic ODEs with neural network residuals.
    
    The model structure follows:
        dx/dt = f_physio(t, x; θ) + g_NN(t, x, GLP1, tVNS; φ)
        
    where f_physio is the mechanistic ODE system and g_NN is the learned residual.
    """
    
    def __init__(self, 
                 ode_params: Optional[Dict[str, float]] = None,
                 nn_hidden: int = 64,
                 nn_layers: int = 4,
                 use_variational: bool = False,
                 prior_params: Optional[Dict[str, Dict[str, float]]] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the hybrid ODE-NN model.
        
        Args:
            ode_params: Parameters for the mechanistic ODE
            nn_hidden: Hidden dimension for the neural network
            nn_layers: Number of hidden layers in the NN
            use_variational: Whether to use variational inference
            prior_params: Prior parameters for Bayesian inference
            device: Computation device
        """
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_variational = use_variational
        
        # Initialize components
        self.ode_core = ODECore(ode_params).to(self.device)
        self.nn_residual = NNResidual(input_dim=9, hidden_dim=nn_hidden, 
                                     output_dim=6, n_layers=nn_layers).to(self.device)
        
        # State dimension and names
        self.n_states = 6
        self.state_names = ['Glucose', 'Insulin', 'Glucagon', 'GLP1', 'GE', 'FFA']
        
        # Variational parameters if using Bayesian inference
        if use_variational:
            self._setup_variational_inference(prior_params)
        else:
            self.variational_params = None
            
    def _setup_variational_inference(self, prior_params: Optional[Dict[str, Dict[str, float]]]):
        """
        Set up variational parameters for Bayesian inference.
        
        Args:
            prior_params: Prior means and stds for parameters
        """
        # Collect all model parameters
        param_shapes = {}
        param_names = []
        
        # ODE parameters
        for name, param in self.ode_core.named_buffers():
            if name in ['a_GI', 'k_I', 'rho', 'E_max', 'EC_50', 'V_max', 'K_m', 'k_L']:
                param_shapes[f'ode_{name}'] = param.shape
                param_names.append(f'ode_{name}')
        
        # NN parameters
        for name, param in self.nn_residual.named_parameters():
            clean_name = name.replace('.', '_')
            param_shapes[f'nn_{clean_name}'] = param.shape
            param_names.append(f'nn_{clean_name}')
        
        # Extract priors
        prior_means = {}
        prior_stds = {}
        
        if prior_params:
            for name in param_names:
                if name in prior_params:
                    prior_means[name] = prior_params[name].get('mean', 0.0)
                    prior_stds[name] = prior_params[name].get('std', 1.0)
        
        # Initialize variational parameters
        self.variational_params = VariationalParameters(
            param_shapes, prior_means, prior_stds
        ).to(self.device)
    
    def ode_residual(self, t: torch.Tensor, state: torch.Tensor,
                     external_inputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute the combined ODE + NN residual dynamics.
        
        Args:
            t: Current time
            state: Current state vector
            external_inputs: External inputs (meal, tVNS, etc.)
            
        Returns:
            dx_dt: State derivatives
        """
        # Get mechanistic ODE derivatives
        ode_derivatives = self.ode_core(t, state, external_inputs)
        
        # Extract GLP1 and tVNS for NN input
        glp1 = state[..., 3] if state.dim() > 1 else state[3]
        tvns = external_inputs.get('tVNS', torch.zeros_like(t)) if external_inputs else torch.zeros_like(t)
        
        # Get NN residuals
        nn_residuals = self.nn_residual(t, state, glp1, tvns)
        
        # Combine ODE and NN contributions
        dx_dt = ode_derivatives + nn_residuals
        
        return dx_dt
    
    def forward(self, initial_state: torch.Tensor, t_span: torch.Tensor,
                external_inputs: Optional[Dict[str, torch.Tensor]] = None,
                solver: str = 'dopri5', rtol: float = 1e-6, atol: float = 1e-8) -> torch.Tensor:
        """
        Solve the hybrid ODE system forward in time.
        
        Args:
            initial_state: Initial state vector (batch_size, n_states) or (n_states,)
            t_span: Time points to evaluate solution (n_time_points,)
            external_inputs: External inputs over time
            solver: ODE solver to use ('dopri5', 'rk45', 'dop853', 'radau', 'bdf')
            rtol: Relative tolerance for solver
            atol: Absolute tolerance for solver
            
        Returns:
            trajectories: Solution trajectories (batch_size, n_time_points, n_states)
        """
        # Handle batch dimension
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = initial_state.shape[0]
        
        # Handle time dimension properly
        if t_span.dim() == 2:
            # Batched time points
            n_time_points = t_span.shape[1]
        else:
            n_time_points = len(t_span)
        
        # Initialize output tensor
        trajectories = torch.zeros(batch_size, n_time_points, self.n_states, 
                                 device=self.device)
        
        # Map solver names from torchdiffeq to scipy
        solver_map = {
            'dopri5': 'DOP853',  # Use DOP853 as it's similar to dopri5
            'rk45': 'RK45',
            'dop853': 'DOP853',
            'radau': 'Radau',
            'bdf': 'BDF'
        }
        scipy_solver = solver_map.get(solver.lower(), solver)
        
        # Solve for each batch element
        for b in range(batch_size):
            # Extract initial conditions
            y0 = initial_state[b].detach().cpu().numpy()
            t_eval = t_span.detach().cpu().numpy()
            
            # Debug: check dimensions
            if t_eval.ndim > 1:
                # Handle batch dimension in time points
                if t_eval.shape[0] == batch_size:
                    t_eval = t_eval[b]
                else:
                    t_eval = t_eval.squeeze()
            
            # Ensure t_eval is 1D
            if t_eval.ndim > 1:
                t_eval = t_eval.flatten()
            
            # Extract scalar values for time span
            t0 = float(t_eval[0])
            tf = float(t_eval[-1])
            
            # Define ODE function for scipy
            def ode_func(t, y):
                t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
                
                # Interpolate external inputs if provided
                ext_inputs = None
                if external_inputs:
                    ext_inputs = {}
                    # Convert time to numpy for searchsorted
                    t_np = t_tensor.cpu().numpy() if isinstance(t_tensor, torch.Tensor) else t
                    
                    for key, values in external_inputs.items():
                        if values.dim() == 2:  # Time-varying input
                            # Linear interpolation
                            idx = np.searchsorted(t_eval, t_np)
                            if idx == 0:
                                ext_inputs[key] = values[b, 0]
                            elif idx >= len(t_eval):
                                ext_inputs[key] = values[b, -1]
                            else:
                                t1, t2 = t_eval[idx-1], t_eval[idx]
                                v1, v2 = values[b, idx-1], values[b, idx]
                                alpha = (t_np - t1) / (t2 - t1)
                                ext_inputs[key] = v1 + alpha * (v2 - v1)
                        else:
                            ext_inputs[key] = values[b]
                
                # Compute derivatives
                with torch.no_grad():
                    dydt = self.ode_residual(t_tensor, y_tensor, ext_inputs)
                
                return dydt.detach().cpu().numpy()
            
            # Solve ODE
            sol = solve_ivp(ode_func, (t0, tf), y0,
                          t_eval=t_eval, method=scipy_solver, rtol=rtol, atol=atol)
            
            if not sol.success:
                logger.warning(f"ODE solver failed for batch {b}: {sol.message}")
            
            # Store solution
            # sol.y has shape (n_states, n_time_points), we need (n_time_points, n_states)
            solution = torch.tensor(sol.y.T, dtype=torch.float32, device=self.device)
            
            # If solution has fewer time points than expected, we need to handle this
            if solution.shape[0] != n_time_points:
                logger.warning(f"Solution has {solution.shape[0]} time points, expected {n_time_points}")
                # For now, just use what we got
                trajectories[b, :solution.shape[0]] = solution
            else:
                trajectories[b] = solution
        
        if squeeze_output:
            trajectories = trajectories.squeeze(0)
            
        return trajectories
    
    def loss(self, batch: Dict[str, torch.Tensor], 
             lambda1: float = 1.0, lambda2: float = 1.0,
             use_physics_loss: bool = True) -> torch.Tensor:
        """
        Compute the combined loss function.
        
        Loss = data_loss + λ1 * physics_loss + λ2 * regularization
        
        Args:
            batch: Dictionary containing:
                - 'initial_state': Initial conditions
                - 'observations': Observed trajectories
                - 'time_points': Time points of observations
                - 'external_inputs': External inputs (optional)
            lambda1: Weight for physics loss
            lambda2: Weight for regularization/Bayesian loss
            use_physics_loss: Whether to include physics constraint loss
            
        Returns:
            total_loss: Combined loss value
        """
        # Extract batch data
        initial_state = batch['initial_state']
        observations = batch['observations']
        time_points = batch['time_points']
        external_inputs = batch.get('external_inputs', None)
        
        # Forward pass
        predictions = self.forward(initial_state, time_points, external_inputs)
        
        # Data loss (MSE)
        data_loss = F.mse_loss(predictions, observations)
        
        # Physics loss (ODE residual matching)
        physics_loss = torch.tensor(0.0, device=self.device)
        if use_physics_loss and lambda1 > 0:
            # Sample time points for physics constraints
            n_physics_points = min(20, len(time_points))
            physics_indices = torch.randperm(len(time_points))[:n_physics_points]
            
            for idx in physics_indices:
                t = time_points[:, idx] if time_points.dim() == 2 else time_points[idx]
                state = predictions[:, idx, :]
                
                # Extract external inputs for this time point
                ext_inputs_t = None
                if external_inputs:
                    ext_inputs_t = {}
                    for key, values in external_inputs.items():
                        if values.dim() == 2:  # Time-varying
                            ext_inputs_t[key] = values[:, idx]
                        else:
                            ext_inputs_t[key] = values
                
                # Compute derivatives using ODE
                with torch.enable_grad():
                    state.requires_grad = True
                    next_state = self.forward(state, torch.tensor([0.0, 0.1]).to(self.device),
                                            ext_inputs_t)[:, 1, :]
                    
                    # Finite difference approximation
                    dx_dt_fd = (next_state - state) / 0.1
                    
                    # ODE derivatives
                    dx_dt_ode = self.ode_residual(t, state, ext_inputs_t)
                    
                    # Physics loss
                    physics_loss += F.mse_loss(dx_dt_fd, dx_dt_ode)
            
            physics_loss = physics_loss / n_physics_points
        
        # Regularization/Bayesian loss
        reg_loss = torch.tensor(0.0, device=self.device)
        if lambda2 > 0:
            if self.use_variational:
                # Bayesian loss (negative ELBO)
                reg_loss = bayes_loss(self, observations, noise_sigma=1.0, n_samples=5)
            else:
                # L2 regularization
                reg_loss = self.nn_residual.regularization_loss(l2_weight=lambda2)
        
        # Total loss
        total_loss = data_loss + lambda1 * physics_loss + lambda2 * reg_loss
        
        # Log components
        logger.debug(f"Loss components - Data: {data_loss:.4f}, "
                    f"Physics: {physics_loss:.4f}, Reg: {reg_loss:.4f}")
        
        return total_loss
    
    def get_variational_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get flattened variational parameters.
        
        Returns:
            mu: Mean parameters
            log_sigma: Log standard deviation parameters
        """
        if not self.use_variational:
            raise ValueError("Model was not initialized with variational inference")
        
        return self.variational_params.get_flattened_params()
    
    def sample_posterior(self, n_samples: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        Sample from the posterior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            samples: List of parameter dictionaries
        """
        if not self.use_variational:
            raise ValueError("Model was not initialized with variational inference")
        
        return self.variational_params.sample(n_samples)
    
    def forward_with_params(self, params: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                           *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with specific parameters (for Bayesian inference).
        
        Args:
            params: Either flattened parameter vector or dictionary
            *args, **kwargs: Arguments for forward()
            
        Returns:
            output: Model output with given parameters
        """
        # Save current parameters
        saved_params = {}
        
        # Set new parameters
        if isinstance(params, torch.Tensor):
            # Flattened vector - need to unflatten
            # This is a simplified version - full implementation would need proper mapping
            logger.warning("Flattened parameter vector not fully implemented")
        else:
            # Dictionary of parameters
            for name, value in params.items():
                # Ensure value is on the correct device
                value = value.to(self.device)
                
                if name.startswith('ode_'):
                    param_name = name[4:]  # Remove 'ode_' prefix
                    if hasattr(self.ode_core, param_name):
                        saved_params[name] = getattr(self.ode_core, param_name).clone()
                        setattr(self.ode_core, param_name, value)
                elif name.startswith('nn_'):
                    # Extract the clean parameter name
                    param_name = name[3:]  # Remove 'nn_' prefix
                    # Set NN parameters by iterating through named parameters
                    for nn_name, nn_param in self.nn_residual.named_parameters():
                        if nn_name.replace('.', '_') == param_name:
                            saved_params[f'nn_{nn_name}'] = nn_param.data.clone()
                            nn_param.data = value.data
                            break
        
        # Forward pass
        output = self.forward(*args, **kwargs)
        
        # Restore parameters
        for name, value in saved_params.items():
            if name.startswith('ode_'):
                param_name = name[4:]
                setattr(self.ode_core, param_name, value)
            elif name.startswith('nn_'):
                # Restore NN parameters
                nn_name = name[3:]  # Remove 'nn_' prefix
                for param_name, param in self.nn_residual.named_parameters():
                    if param_name == nn_name:
                        param.data = value
                        break
        
        return output