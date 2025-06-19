"""
MCMC sampling using NUTS (No-U-Turn Sampler) for the hybrid ODE-NN model.

Provides exact Bayesian inference as an alternative to variational inference.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import arviz as az
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_nuts(model, data: Dict[str, torch.Tensor], 
             num_samples: int = 1000, num_warmup: int = 500,
             target_accept: float = 0.8, max_tree_depth: int = 10,
             device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    """
    Run NUTS sampler for Bayesian inference.
    
    This is a lightweight implementation placeholder. For production use,
    consider using PyMC3, Stan, or NumPyro.
    
    Args:
        model: HybridODENN model
        data: Dictionary containing observations and inputs
        num_samples: Number of MCMC samples to draw
        num_warmup: Number of warmup/burn-in samples
        target_accept: Target acceptance probability
        max_tree_depth: Maximum tree depth for NUTS
        device: Computation device
        
    Returns:
        samples: Dictionary of posterior samples
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Running NUTS sampler: {num_samples} samples, {num_warmup} warmup")
    
    # Extract data
    initial_state = data['initial_state'].to(device)
    observations = data['observations'].to(device)
    time_points = data['time_points'].to(device)
    external_inputs = data.get('external_inputs', None)
    
    # Define log probability function
    def log_prob_fn(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute log posterior probability.
        
        log p(θ|x) ∝ log p(x|θ) + log p(θ)
        """
        # Prior log probability
        log_prior = 0.0
        
        # ODE parameters have informative priors
        ode_param_priors = {
            'a_GI': {'mean': 0.0104, 'std': 0.002},
            'k_I': {'mean': 0.025, 'std': 0.005},
            'rho': {'mean': 0.003, 'std': 0.001},
            'E_max': {'mean': 0.1, 'std': 0.02},
            'V_max': {'mean': 9.0, 'std': 2.0},
            'K_m': {'mean': 7.0, 'std': 1.5},
            'k_L': {'mean': 0.02, 'std': 0.005}
        }
        
        for name, prior in ode_param_priors.items():
            if f'ode.{name}' in params:
                value = params[f'ode.{name}']
                # Gaussian prior
                log_prior += -0.5 * ((value - prior['mean']) / prior['std'])**2
                log_prior -= 0.5 * np.log(2 * np.pi * prior['std']**2)
        
        # NN parameters have weakly informative priors (N(0, 1))
        for name, value in params.items():
            if name.startswith('nn.'):
                log_prior += -0.5 * (value**2).sum()
                log_prior -= 0.5 * value.numel() * np.log(2 * np.pi)
        
        # Likelihood
        try:
            # Forward pass with current parameters
            predictions = model.forward_with_params(
                params, initial_state, time_points, external_inputs
            )
            
            # Gaussian likelihood with fixed noise
            noise_sigma = 1.0
            log_likelihood = -0.5 * ((observations - predictions) / noise_sigma)**2
            log_likelihood = log_likelihood.sum()
            log_likelihood -= 0.5 * observations.numel() * np.log(2 * np.pi * noise_sigma**2)
            
        except Exception as e:
            logger.warning(f"Error in forward pass: {e}")
            return torch.tensor(-float('inf'))
        
        return log_prior + log_likelihood
    
    # Initialize parameters
    current_params = {}
    
    # Initialize ODE parameters near their priors
    for name in ['a_GI', 'k_I', 'rho', 'E_max', 'V_max', 'K_m', 'k_L']:
        if hasattr(model.ode_core, name):
            current_value = getattr(model.ode_core, name).clone()
            current_params[f'ode.{name}'] = current_value + 0.01 * torch.randn_like(current_value)
    
    # Initialize NN parameters
    for name, param in model.nn_residual.named_parameters():
        current_params[f'nn.{name}'] = param.clone() + 0.01 * torch.randn_like(param)
    
    # Storage for samples
    samples = {name: [] for name in current_params.keys()}
    
    # NUTS sampler state
    step_size = 0.01
    accept_count = 0
    
    # Combined warmup and sampling
    total_iterations = num_warmup + num_samples
    
    with tqdm(total=total_iterations, desc="MCMC sampling") as pbar:
        for iteration in range(total_iterations):
            # Simple Metropolis-Hastings step (placeholder for full NUTS)
            # In practice, use a proper NUTS implementation
            
            # Propose new parameters
            proposed_params = {}
            for name, value in current_params.items():
                proposed_params[name] = value + step_size * torch.randn_like(value)
            
            # Compute acceptance ratio
            current_log_prob = log_prob_fn(current_params)
            proposed_log_prob = log_prob_fn(proposed_params)
            
            log_accept_ratio = proposed_log_prob - current_log_prob
            
            # Accept or reject
            if torch.log(torch.rand(1)) < log_accept_ratio:
                current_params = proposed_params
                accept_count += 1
            
            # Adapt step size during warmup
            if iteration < num_warmup:
                accept_rate = accept_count / (iteration + 1)
                if accept_rate < target_accept - 0.1:
                    step_size *= 0.9
                elif accept_rate > target_accept + 0.1:
                    step_size *= 1.1
            
            # Store samples after warmup
            if iteration >= num_warmup:
                for name, value in current_params.items():
                    samples[name].append(value.detach().cpu().numpy())
            
            # Update progress bar
            pbar.update(1)
            if iteration % 100 == 0:
                accept_rate = accept_count / (iteration + 1)
                pbar.set_postfix({'accept_rate': f"{accept_rate:.3f}", 
                                 'step_size': f"{step_size:.3e}"})
    
    # Convert to numpy arrays
    for name in samples:
        samples[name] = np.array(samples[name])
    
    # Log summary statistics
    accept_rate = accept_count / total_iterations
    logger.info(f"MCMC completed. Acceptance rate: {accept_rate:.3f}")
    
    # Compute effective sample size
    for name, values in samples.items():
        if values.ndim == 1:
            ess = compute_ess(values)
            logger.info(f"  {name}: ESS = {ess:.0f}")
    
    return samples


def compute_ess(x: np.ndarray) -> float:
    """
    Compute effective sample size using autocorrelation.
    
    Args:
        x: MCMC samples (n_samples,)
        
    Returns:
        ess: Effective sample size
    """
    n = len(x)
    
    # Compute autocorrelation
    x_centered = x - np.mean(x)
    c0 = np.var(x)
    
    # Compute autocorrelation at each lag
    acf = []
    for k in range(min(n // 4, 100)):  # Limit lag to n/4
        ck = np.mean(x_centered[:-k-1] * x_centered[k+1:]) if k > 0 else c0
        acf.append(ck / c0)
    
    # Find first negative autocorrelation
    sum_acf = 0
    for k, rho_k in enumerate(acf):
        if k > 0 and rho_k < 0:
            break
        sum_acf += rho_k
    
    # ESS = n / (1 + 2 * sum of autocorrelations)
    ess = n / (1 + 2 * sum_acf)
    
    return ess


def posterior_summary(samples: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute posterior summary statistics.
    
    Args:
        samples: Dictionary of MCMC samples
        
    Returns:
        summary: Dictionary with mean, std, and quantiles
    """
    summary = {}
    
    for name, values in samples.items():
        if values.ndim == 1:
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'q025': np.percentile(values, 2.5),
                'q975': np.percentile(values, 97.5)
            }
        else:
            # For multi-dimensional parameters, compute element-wise
            summary[name] = {
                'mean': np.mean(values, axis=0),
                'std': np.std(values, axis=0),
                'median': np.median(values, axis=0),
                'q025': np.percentile(values, 2.5, axis=0),
                'q975': np.percentile(values, 97.5, axis=0)
            }
    
    return summary


def save_mcmc_results(samples: Dict[str, np.ndarray], 
                     path: str, metadata: Optional[Dict] = None):
    """
    Save MCMC results to file.
    
    Args:
        samples: Dictionary of MCMC samples
        path: Output file path
        metadata: Additional metadata to save
    """
    # Convert to ArviZ InferenceData for better analysis
    coords = {}
    dims = {}
    
    # Reshape samples for ArviZ
    data_vars = {}
    for name, values in samples.items():
        if values.ndim == 1:
            data_vars[name] = values.reshape(1, -1)  # (chain, draw)
        else:
            # Handle multi-dimensional parameters
            shape = values.shape
            data_vars[name] = values.reshape(1, shape[0], -1)  # (chain, draw, ...)
    
    # Create InferenceData
    posterior = az.from_dict(
        posterior=data_vars,
        coords=coords,
        dims=dims
    )
    
    # Add metadata as attributes
    if metadata:
        for key, value in metadata.items():
            posterior.posterior.attrs[key] = value
    
    # Save to NetCDF
    posterior.to_netcdf(path)
    logger.info(f"MCMC results saved to {path}")


def load_mcmc_results(path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load MCMC results from file.
    
    Args:
        path: Input file path
        
    Returns:
        samples: Dictionary of MCMC samples
        metadata: Metadata dictionary
    """
    # Load InferenceData
    idata = az.from_netcdf(path)
    
    # Extract samples
    samples = {}
    for var_name in idata.posterior.data_vars:
        values = idata.posterior[var_name].values
        # Remove chain dimension if single chain
        if values.shape[0] == 1:
            values = values[0]
        samples[var_name] = values
    
    # Extract metadata
    metadata = dict(idata.posterior.attrs)
    
    return samples, metadata