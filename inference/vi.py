"""
Variational inference implementation for the hybrid ODE-NN model.

Implements mean-field variational inference with reparameterization trick
for efficient gradient-based optimization.
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VariationalInference:
    """
    Variational inference trainer for hybrid ODE-NN models.
    
    Uses diagonal Gaussian variational family with reparameterization trick.
    """
    
    def __init__(self, model, prior_params: Optional[Dict[str, Dict[str, float]]] = None,
                 learning_rate: float = 1e-3, device: Optional[torch.device] = None):
        """
        Initialize variational inference trainer.
        
        Args:
            model: HybridODENN model instance
            prior_params: Prior parameters for each model parameter
            learning_rate: Learning rate for variational parameters
            device: Computation device
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        
        # Ensure model has variational parameters
        if not hasattr(model, 'variational_params') or model.variational_params is None:
            raise ValueError("Model must be initialized with use_variational=True")
        
        self.variational_params = model.variational_params
        
        # Set up optimizer for variational parameters
        self.optimizer = torch.optim.Adam(
            self.variational_params.parameters(),
            lr=learning_rate
        )
        
        # Training history
        self.history = {
            'elbo': [],
            'kl': [],
            'log_likelihood': []
        }
    
    def elbo(self, batch: Dict[str, torch.Tensor], n_samples: int = 5,
             noise_sigma: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Evidence Lower Bound (ELBO).
        
        ELBO = E_q[log p(x|ψ)] - KL[q(ψ) || p(ψ)]
        
        Args:
            batch: Data batch containing observations and inputs
            n_samples: Number of Monte Carlo samples
            noise_sigma: Observation noise standard deviation
            
        Returns:
            elbo: ELBO value (to maximize)
            components: Dictionary with ELBO components
        """
        # Extract batch data
        initial_state = batch['initial_state']
        observations = batch['observations']
        time_points = batch['time_points']
        external_inputs = batch.get('external_inputs', None)
        
        # Compute KL divergence
        kl_div = self.variational_params.kl_divergence()
        
        # Monte Carlo estimate of expected log-likelihood
        log_likelihood = 0.0
        
        for _ in range(n_samples):
            # Sample parameters from variational posterior
            param_samples = self.variational_params.sample(1)[0]
            
            # Forward pass with sampled parameters
            predictions = self.model.forward_with_params(
                param_samples, initial_state, time_points, external_inputs
            )
            
            # Compute log-likelihood
            # log p(x|ψ) = -0.5 * sum((x - x̂)² / σ²) - 0.5 * n * log(2π σ²)
            squared_error = ((observations - predictions) / noise_sigma).pow(2).sum()
            log_likelihood += -0.5 * squared_error
        
        # Average over samples
        log_likelihood = log_likelihood / n_samples
        
        # Add constant term
        n_obs = observations.numel()
        log_likelihood -= 0.5 * n_obs * np.log(2 * np.pi * noise_sigma**2)
        
        # ELBO = log_likelihood - kl_div
        elbo = log_likelihood - kl_div
        
        components = {
            'elbo': elbo,
            'kl': kl_div,
            'log_likelihood': log_likelihood
        }
        
        return elbo, components
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   n_samples: int = 5) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Data batch
            n_samples: Number of MC samples for ELBO estimation
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.optimizer.zero_grad()
        
        # Compute ELBO (negative because we minimize)
        elbo, components = self.elbo(batch, n_samples=n_samples)
        loss = -elbo
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.variational_params.parameters(), max_norm=5.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Record metrics
        metrics = {
            'loss': loss.item(),
            'elbo': elbo.item(),
            'kl': components['kl'].item(),
            'log_likelihood': components['log_likelihood'].item()
        }
        
        return metrics
    
    def train(self, train_loader, val_loader=None, epochs: int = 100,
              n_samples: int = 5, early_stopping_patience: int = 10,
              verbose: bool = True):
        """
        Train the variational model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            n_samples: Number of MC samples for ELBO
            early_stopping_patience: Patience for early stopping
            verbose: Whether to show progress
        """
        best_val_elbo = -float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_metrics = {'loss': 0, 'elbo': 0, 'kl': 0, 'log_likelihood': 0}
            
            if verbose:
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            else:
                pbar = train_loader
            
            for batch in pbar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Training step
                metrics = self.train_step(batch, n_samples=n_samples)
                
                # Accumulate metrics
                for key in train_metrics:
                    train_metrics[key] += metrics[key]
                
                if verbose:
                    pbar.set_postfix({'ELBO': f"{metrics['elbo']:.4f}"})
            
            # Average training metrics
            n_batches = len(train_loader)
            for key in train_metrics:
                train_metrics[key] /= n_batches
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_metrics = {'loss': 0, 'elbo': 0, 'kl': 0, 'log_likelihood': 0}
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Move batch to device
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(self.device)
                        
                        # Compute ELBO
                        elbo, components = self.elbo(batch, n_samples=n_samples)
                        
                        # Accumulate metrics
                        val_metrics['loss'] += (-elbo).item()
                        val_metrics['elbo'] += elbo.item()
                        val_metrics['kl'] += components['kl'].item()
                        val_metrics['log_likelihood'] += components['log_likelihood'].item()
                
                # Average validation metrics
                n_val_batches = len(val_loader)
                for key in val_metrics:
                    val_metrics[key] /= n_val_batches
                
                # Early stopping
                if val_metrics['elbo'] > best_val_elbo:
                    best_val_elbo = val_metrics['elbo']
                    patience_counter = 0
                    # Save best model
                    self.best_state = self.variational_params.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Update history
            self.history['elbo'].append(train_metrics['elbo'])
            self.history['kl'].append(train_metrics['kl'])
            self.history['log_likelihood'].append(train_metrics['log_likelihood'])
            
            # Log progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: "
                          f"Train ELBO={train_metrics['elbo']:.4f}, "
                          f"KL={train_metrics['kl']:.4f}, "
                          f"LL={train_metrics['log_likelihood']:.4f}")
                if val_loader is not None:
                    logger.info(f"  Val ELBO={val_metrics['elbo']:.4f}")
        
        # Restore best model if using validation
        if val_loader is not None and hasattr(self, 'best_state'):
            self.variational_params.load_state_dict(self.best_state)
    
    def sample_posterior(self, n_samples: int = 100) -> List[Dict[str, torch.Tensor]]:
        """
        Sample from the learned posterior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            samples: List of parameter dictionaries
        """
        return self.variational_params.sample(n_samples)
    
    def posterior_predictive(self, initial_state: torch.Tensor,
                           time_points: torch.Tensor,
                           external_inputs: Optional[Dict[str, torch.Tensor]] = None,
                           n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior predictive distribution.
        
        Args:
            initial_state: Initial conditions
            time_points: Time points for prediction
            external_inputs: External inputs
            n_samples: Number of posterior samples
            
        Returns:
            mean: Posterior predictive mean
            std: Posterior predictive standard deviation
        """
        predictions = []
        
        # Sample from posterior and make predictions
        for _ in range(n_samples):
            # Sample parameters
            param_sample = self.variational_params.sample(1)[0]
            
            # Make prediction
            with torch.no_grad():
                pred = self.model.forward_with_params(
                    param_sample, initial_state, time_points, external_inputs
                )
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute statistics
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'variational_params': self.variational_params.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.variational_params.load_state_dict(checkpoint['variational_params'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {path}")