"""
Bayesian inference utilities for the hybrid ODE-NN model.

Implements variational inference and MCMC sampling for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
import numpy as np


def bayes_loss(model, x_obs: torch.Tensor, noise_sigma: float = 1.0, 
               n_samples: int = 5) -> torch.Tensor:
    """
    Compute the negative ELBO (Evidence Lower Bound) for variational inference.
    
    Args:
        model: HybridODENN model with variational parameters
        x_obs: Observed data tensor (batch_size, n_states, n_time_points)
        noise_sigma: Observation noise standard deviation
        n_samples: Number of Monte Carlo samples for likelihood estimation
        
    Returns:
        neg_elbo: Negative ELBO (scalar tensor)
    """
    # Get variational parameters
    mu, log_sigma = model.get_variational_params()
    
    # Compute KL divergence from prior
    # KL[q(ψ|η) || p(ψ)] = 0.5 * sum(σ² + μ² - 1 - 2*log(σ))
    kl = 0.5 * (log_sigma.exp().pow(2) + mu.pow(2) - 1 - 2 * log_sigma).sum()
    
    # Monte Carlo estimate of expected log-likelihood
    log_likelihood = 0.0
    
    for _ in range(n_samples):
        # Reparameterization trick: ψ = μ + σ * ε, where ε ~ N(0, I)
        eps = torch.randn_like(mu)
        psi = mu + eps * log_sigma.exp()
        
        # Forward pass with sampled parameters
        x_hat = model.forward_with_params(psi, x_obs)
        
        # Gaussian log-likelihood
        # log p(x|ψ) = -0.5 * sum((x - x̂)² / σ²) - 0.5 * n * log(2π σ²)
        squared_error = ((x_obs - x_hat) / noise_sigma).pow(2).sum()
        log_likelihood += -0.5 * squared_error
    
    # Average over samples
    log_likelihood = log_likelihood / n_samples
    
    # Add constant term
    n_obs = x_obs.numel()
    log_likelihood -= 0.5 * n_obs * math.log(2 * math.pi * noise_sigma**2)
    
    # Negative ELBO = KL - E[log p(x|ψ)]
    neg_elbo = kl - log_likelihood
    
    return neg_elbo


class VariationalParameters(nn.Module):
    """
    Variational parameters for diagonal Gaussian posterior.
    
    Maintains mean and log-standard deviation for each model parameter.
    """
    
    def __init__(self, param_shapes: Dict[str, torch.Size], 
                 prior_means: Optional[Dict[str, float]] = None,
                 prior_stds: Optional[Dict[str, float]] = None):
        """
        Initialize variational parameters.
        
        Args:
            param_shapes: Dictionary mapping parameter names to shapes
            prior_means: Prior means for each parameter (default: 0)
            prior_stds: Prior standard deviations (default: 1)
        """
        super().__init__()
        
        self.param_shapes = param_shapes
        self.prior_means = prior_means or {}
        self.prior_stds = prior_stds or {}
        
        # Initialize variational parameters
        self.means = nn.ParameterDict()
        self.log_stds = nn.ParameterDict()
        
        for name, shape in param_shapes.items():
            # Initialize mean from prior or zero
            prior_mean = self.prior_means.get(name, 0.0)
            self.means[name] = nn.Parameter(torch.full(shape, prior_mean))
            
            # Initialize log std (start with small uncertainty)
            prior_std = self.prior_stds.get(name, 1.0)
            init_log_std = math.log(prior_std * 0.1)  # Start with 10% of prior std
            self.log_stds[name] = nn.Parameter(torch.full(shape, init_log_std))
    
    def sample(self, n_samples: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        Sample from the variational posterior.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            samples: List of parameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name in self.param_shapes:
                mean = self.means[name]
                log_std = self.log_stds[name]
                
                # Reparameterization trick
                eps = torch.randn_like(mean)
                sample[name] = mean + eps * log_std.exp()
            
            samples.append(sample)
        
        return samples
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence from prior.
        
        Returns:
            kl: KL[q(ψ|η) || p(ψ)] summed over all parameters
        """
        kl = 0.0
        
        for name in self.param_shapes:
            mean = self.means[name]
            log_std = self.log_stds[name]
            
            # Get prior parameters
            prior_mean = self.prior_means.get(name, 0.0)
            prior_std = self.prior_stds.get(name, 1.0)
            prior_log_std = math.log(prior_std)
            
            # KL divergence between two Gaussians
            # KL[N(μ1,σ1) || N(μ2,σ2)] = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
            kl_term = (prior_log_std - log_std + 
                      (log_std.exp().pow(2) + (mean - prior_mean).pow(2)) / 
                      (2 * prior_std**2) - 0.5)
            
            kl += kl_term.sum()
        
        return kl
    
    def get_flattened_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get flattened mean and log-std vectors.
        
        Returns:
            mu: Flattened mean vector
            log_sigma: Flattened log-std vector
        """
        mu_list = []
        log_sigma_list = []
        
        for name in sorted(self.param_shapes.keys()):  # Sort for consistency
            mu_list.append(self.means[name].flatten())
            log_sigma_list.append(self.log_stds[name].flatten())
        
        mu = torch.cat(mu_list)
        log_sigma = torch.cat(log_sigma_list)
        
        return mu, log_sigma


def compute_posterior_predictive(model, x_initial: torch.Tensor, 
                                t_span: torch.Tensor,
                                external_inputs: Optional[Dict[str, torch.Tensor]] = None,
                                n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute posterior predictive distribution.
    
    Args:
        model: Trained model with variational parameters
        x_initial: Initial state
        t_span: Time points for prediction
        external_inputs: External inputs (meals, etc.)
        n_samples: Number of posterior samples
        
    Returns:
        mean: Posterior predictive mean (n_time, n_states)
        std: Posterior predictive std (n_time, n_states)
    """
    predictions = []
    
    # Sample from posterior and make predictions
    for _ in range(n_samples):
        # Sample parameters
        params = model.sample_posterior()
        
        # Make prediction with sampled parameters
        with torch.no_grad():
            pred = model.forward_with_params(params, x_initial, t_span, external_inputs)
            predictions.append(pred)
    
    # Stack predictions
    predictions = torch.stack(predictions)  # (n_samples, n_time, n_states)
    
    # Compute mean and std
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    return mean, std