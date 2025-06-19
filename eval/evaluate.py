"""
Evaluation metrics for the hybrid ODE-NN model.

Computes RMSE, MAE, calibration error, and other metrics for model assessment.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from pathlib import Path
import logging
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_ode_nn import HybridODENN
from inference.vi import VariationalInference

logger = logging.getLogger(__name__)


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor, 
                 per_state: bool = False) -> Union[float, np.ndarray]:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predictions: Model predictions (batch, time, states)
        targets: Ground truth values (batch, time, states)
        per_state: Whether to return RMSE per state variable
        
    Returns:
        rmse: Overall RMSE or per-state RMSE array
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    if per_state:
        # Compute RMSE for each state variable
        n_states = predictions.shape[-1]
        rmse_per_state = np.zeros(n_states)
        
        for i in range(n_states):
            rmse_per_state[i] = np.sqrt(mean_squared_error(
                targets[..., i].flatten(),
                predictions[..., i].flatten()
            ))
        
        return rmse_per_state
    else:
        # Overall RMSE
        return np.sqrt(mean_squared_error(
            targets.flatten(),
            predictions.flatten()
        ))


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor,
                per_state: bool = False) -> Union[float, np.ndarray]:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Model predictions (batch, time, states)
        targets: Ground truth values (batch, time, states)
        per_state: Whether to return MAE per state variable
        
    Returns:
        mae: Overall MAE or per-state MAE array
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    if per_state:
        # Compute MAE for each state variable
        n_states = predictions.shape[-1]
        mae_per_state = np.zeros(n_states)
        
        for i in range(n_states):
            mae_per_state[i] = mean_absolute_error(
                targets[..., i].flatten(),
                predictions[..., i].flatten()
            )
        
        return mae_per_state
    else:
        # Overall MAE
        return mean_absolute_error(
            targets.flatten(),
            predictions.flatten()
        )


def compute_calibration_error(predictions: torch.Tensor, 
                            uncertainties: torch.Tensor,
                            targets: torch.Tensor,
                            n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration metrics for uncertainty quantification.
    
    Expected Calibration Error (ECE) measures how well the predicted
    uncertainties match the actual errors.
    
    Args:
        predictions: Mean predictions (batch, time, states)
        uncertainties: Predicted standard deviations (batch, time, states)
        targets: Ground truth values (batch, time, states)
        n_bins: Number of bins for calibration
        
    Returns:
        metrics: Dictionary containing ECE and other calibration metrics
    """
    predictions = predictions.detach().cpu().numpy()
    uncertainties = uncertainties.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    unc_flat = uncertainties.flatten()
    target_flat = targets.flatten()
    
    # Compute normalized errors
    errors = np.abs(pred_flat - target_flat)
    normalized_errors = errors / (unc_flat + 1e-6)
    
    # Compute calibration curve
    confidence_levels = np.linspace(0, 1, n_bins + 1)
    expected_freq = []
    observed_freq = []
    
    for i in range(n_bins):
        # For each confidence level, check if error is within predicted bounds
        conf = confidence_levels[i]
        z_score = np.abs(np.random.randn(10000))  # Standard normal
        threshold = np.percentile(z_score, conf * 100)
        
        # Count how many normalized errors are below threshold
        mask = normalized_errors <= threshold
        if len(mask) > 0:
            observed = np.mean(mask)
            expected = conf
            
            expected_freq.append(expected)
            observed_freq.append(observed)
    
    # Expected Calibration Error
    ece = np.mean(np.abs(np.array(expected_freq) - np.array(observed_freq)))
    
    # Mean Scaled Interval Score (MSIS)
    # Measures quality of prediction intervals
    alpha = 0.05  # 95% prediction interval
    z_alpha = 1.96  # For 95% interval
    
    lower = pred_flat - z_alpha * unc_flat
    upper = pred_flat + z_alpha * unc_flat
    interval_width = upper - lower
    
    # Penalty for observations outside interval
    penalty = 2 / alpha * (
        (target_flat < lower) * (lower - target_flat) +
        (target_flat > upper) * (target_flat - upper)
    )
    
    msis = np.mean(interval_width + penalty)
    
    # Sharpness: average prediction uncertainty
    sharpness = np.mean(unc_flat)
    
    # Coverage: fraction of targets within prediction intervals
    coverage = np.mean((target_flat >= lower) & (target_flat <= upper))
    
    return {
        'ece': ece,
        'msis': msis,
        'sharpness': sharpness,
        'coverage_95': coverage,
        'mean_normalized_error': np.mean(normalized_errors)
    }


def evaluate_model(model: HybridODENN, test_loader, device: torch.device,
                   use_variational: bool = False, n_posterior_samples: int = 100) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        use_variational: Whether model uses variational inference
        n_posterior_samples: Number of posterior samples for uncertainty
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    state_names = ['Glucose', 'Insulin', 'Glucagon', 'GLP1', 'GE', 'FFA']
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for k, v in batch[key].items():
                        batch[key][k] = v.to(device)
            
            initial_state = batch['initial_state']
            targets = batch['observations']
            time_points = batch['time_points']
            external_inputs = batch.get('external_inputs', None)
            
            if use_variational and hasattr(model, 'variational_params'):
                # Compute posterior predictive distribution
                predictions_samples = []
                
                for _ in range(n_posterior_samples):
                    # Sample from posterior
                    param_sample = model.variational_params.sample(1)[0]
                    
                    # Make prediction
                    pred = model.forward_with_params(
                        param_sample, initial_state, time_points, external_inputs
                    )
                    predictions_samples.append(pred)
                
                # Stack samples
                predictions_samples = torch.stack(predictions_samples)
                
                # Compute mean and std
                predictions = predictions_samples.mean(dim=0)
                uncertainties = predictions_samples.std(dim=0)
                
            else:
                # Point estimate
                predictions = model.forward(initial_state, time_points, external_inputs)
                # Use a fixed uncertainty for non-Bayesian models
                uncertainties = torch.ones_like(predictions) * 0.1
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_uncertainties.append(uncertainties)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    
    # Compute metrics
    metrics = {}
    
    # Overall metrics
    metrics['rmse'] = compute_rmse(all_predictions, all_targets)
    metrics['mae'] = compute_mae(all_predictions, all_targets)
    
    # Per-state metrics
    rmse_per_state = compute_rmse(all_predictions, all_targets, per_state=True)
    mae_per_state = compute_mae(all_predictions, all_targets, per_state=True)
    
    for i, state_name in enumerate(state_names):
        metrics[f'rmse_{state_name.lower()}'] = rmse_per_state[i]
        metrics[f'mae_{state_name.lower()}'] = mae_per_state[i]
    
    # Calibration metrics (if using uncertainty)
    if use_variational:
        calibration_metrics = compute_calibration_error(
            all_predictions, all_uncertainties, all_targets
        )
        metrics.update(calibration_metrics)
    
    # Normalized metrics (useful for comparison)
    target_std = all_targets.std(dim=(0, 1)).cpu().numpy()
    metrics['nrmse'] = metrics['rmse'] / np.mean(target_std)
    
    # Per-state normalized RMSE
    for i, state_name in enumerate(state_names):
        metrics[f'nrmse_{state_name.lower()}'] = rmse_per_state[i] / target_std[i]
    
    return metrics


def evaluate_checkpoint(checkpoint_path: str, test_loader, device: torch.device,
                       config: Optional[dict] = None) -> Dict[str, float]:
    """
    Evaluate a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        test_loader: Test data loader
        device: Computation device
        config: Model configuration (if not in checkpoint)
        
    Returns:
        metrics: Evaluation metrics
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Configuration not found in checkpoint")
    
    # Create model
    use_variational = config.get('model', {}).get('use_variational', False)
    model = HybridODENN(
        ode_params=None,
        nn_hidden=config['model'].get('nn_hidden', 64),
        nn_layers=config['model'].get('nn_layers', 4),
        use_variational=use_variational,
        device=device
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, use_variational)
    
    # Add checkpoint info
    metrics['checkpoint_epoch'] = checkpoint.get('epoch', -1)
    metrics['checkpoint_val_loss'] = checkpoint.get('val_loss', -1)
    
    return metrics


def save_evaluation_results(metrics: Dict[str, float], output_path: str):
    """
    Save evaluation results to file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Output file path
    """
    # Convert to DataFrame for nice formatting
    df = pd.DataFrame([metrics])
    
    # Save as CSV
    df.to_csv(output_path, index=False)
    
    # Also save as formatted text
    text_path = Path(output_path).with_suffix('.txt')
    with open(text_path, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {metrics['mae']:.4f}\n")
        f.write(f"  Normalized RMSE: {metrics['nrmse']:.4f}\n")
        f.write("\n")
        
        # Per-state metrics
        f.write("Per-State RMSE:\n")
        state_names = ['Glucose', 'Insulin', 'Glucagon', 'GLP1', 'GE', 'FFA']
        for state in state_names:
            key = f'rmse_{state.lower()}'
            if key in metrics:
                f.write(f"  {state}: {metrics[key]:.4f}\n")
        f.write("\n")
        
        # Calibration metrics
        if 'ece' in metrics:
            f.write("Calibration Metrics:\n")
            f.write(f"  Expected Calibration Error: {metrics['ece']:.4f}\n")
            f.write(f"  95% Coverage: {metrics['coverage_95']:.4f}\n")
            f.write(f"  Sharpness: {metrics['sharpness']:.4f}\n")
            f.write(f"  MSIS: {metrics['msis']:.4f}\n")
    
    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    from train.train_hybrid import create_data_loaders, load_config
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/4gi_dataset.csv',
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint.get('config', None)
        if config is None:
            raise ValueError("Configuration not found. Please provide --config")
    
    # Update data path
    config['data']['path'] = args.data
    
    # Create data loader
    _, _, test_loader = create_data_loaders(config)
    
    # Evaluate
    metrics = evaluate_checkpoint(args.checkpoint, test_loader, device, config)
    
    # Save results
    save_evaluation_results(metrics, args.output)
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Normalized RMSE: {metrics['nrmse']:.4f}")