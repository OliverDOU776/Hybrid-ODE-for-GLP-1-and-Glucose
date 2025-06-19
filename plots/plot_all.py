"""
Generate all figures from the paper.

Creates:
- Figure 1: Time series predictions
- Figure 2: Sensitivity analysis (Sobol indices)
- Figure 3: Posterior predictive bands
- Tables III-V: Performance metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import logging
import sys
from SALib.sample import saltelli
from SALib.analyze import sobol

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_ode_nn import HybridODENN
from train.train_hybrid import GlucoseDataset, create_data_loaders
from eval.evaluate import evaluate_model

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def plot_fig1_predictions(model: HybridODENN, test_loader, device: torch.device,
                         save_path: str = "results/figures/fig1_predictions.png",
                         n_subjects: int = 3):
    """
    Plot Figure 1: Time series predictions for multiple subjects.
    
    Shows model predictions vs ground truth for glucose, insulin, and GLP-1.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        save_path: Path to save figure
        n_subjects: Number of subjects to plot
    """
    model.eval()
    
    fig, axes = plt.subplots(n_subjects, 3, figsize=(15, 3*n_subjects))
    if n_subjects == 1:
        axes = axes.reshape(1, -1)
    
    state_names = ['Glucose (mmol/L)', 'Insulin (pmol/L)', 'GLP-1 (pmol/L)']
    state_indices = [0, 1, 3]  # Glucose, Insulin, GLP-1
    
    subject_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if subject_count >= n_subjects:
                break
                
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for k, v in batch[key].items():
                        batch[key][k] = v.to(device)
            
            # Get predictions
            initial_state = batch['initial_state']
            targets = batch['observations']
            time_points = batch['time_points']
            external_inputs = batch.get('external_inputs', None)
            
            predictions = model.forward(initial_state, time_points, external_inputs)
            
            # Plot first sample in batch
            for i in range(min(predictions.shape[0], n_subjects - subject_count)):
                time = time_points[i].cpu().numpy()
                
                for j, (state_idx, state_name) in enumerate(zip(state_indices, state_names)):
                    ax = axes[subject_count, j]
                    
                    # Plot ground truth
                    ax.plot(time, targets[i, :, state_idx].cpu().numpy(),
                           'k-', label='Ground Truth', linewidth=2)
                    
                    # Plot prediction
                    ax.plot(time, predictions[i, :, state_idx].cpu().numpy(),
                           'r--', label='Prediction', linewidth=2)
                    
                    # Add meal indicators if available
                    if external_inputs and 'meal' in external_inputs:
                        meal_times = time[external_inputs['meal'][i].cpu().numpy() > 0]
                        for meal_time in meal_times:
                            ax.axvline(meal_time, color='green', alpha=0.3, linestyle=':')
                    
                    ax.set_xlabel('Time (hours)')
                    ax.set_ylabel(state_name)
                    ax.set_title(f'Subject {subject_count + 1}')
                    
                    if subject_count == 0 and j == 0:
                        ax.legend()
                
                subject_count += 1
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure 1 saved to {save_path}")


def plot_fig2_sensitivity(model: HybridODENN, device: torch.device,
                         save_path: str = "results/figures/fig2_sensitivity.png"):
    """
    Plot Figure 2: Sensitivity analysis using Sobol indices.
    
    Shows which parameters have the most influence on model outputs.
    
    Args:
        model: Trained model
        device: Computation device
        save_path: Path to save figure
    """
    model.eval()
    
    # Define parameter bounds for sensitivity analysis
    param_names = ['a_GI', 'k_I', 'rho', 'E_max', 'V_max', 'K_m', 'k_L']
    param_bounds = [
        [0.008, 0.012],   # a_GI
        [0.02, 0.03],     # k_I
        [0.002, 0.004],   # rho
        [0.08, 0.12],     # E_max
        [7.0, 11.0],      # V_max
        [5.5, 8.5],       # K_m
        [0.015, 0.025]    # k_L
    ]
    
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': param_bounds
    }
    
    # Generate samples using Saltelli sampling
    n_samples = 1024
    param_samples = saltelli.sample(problem, n_samples)
    
    # Run model for each parameter set
    logger.info("Running sensitivity analysis...")
    
    # Use a standard initial condition and time span
    initial_state = torch.tensor([5.0, 60.0, 80.0, 0.0, 0.0, 1.0], device=device)
    time_points = torch.linspace(0, 5, 61, device=device)
    
    # Outputs to analyze
    output_names = ['Glucose AUC', 'Insulin Peak', 'GLP-1 Response']
    outputs = np.zeros((param_samples.shape[0], len(output_names)))
    
    for i, params in enumerate(param_samples):
        if i % 100 == 0:
            logger.info(f"  Sample {i}/{len(param_samples)}")
        
        # Set model parameters
        param_dict = {name: float(value) for name, value in zip(param_names, params)}
        
        # Update ODE parameters
        for name, value in param_dict.items():
            if hasattr(model.ode_core, name):
                setattr(model.ode_core, name, torch.tensor(value, device=device))
        
        # Run simulation with meal input
        meal = torch.zeros(61, device=device)
        meal[6] = 75.0  # 75 mmol glucose at 30 min
        
        external_inputs = {'meal': meal.unsqueeze(0), 'tVNS': torch.zeros(61, device=device).unsqueeze(0)}
        
        with torch.no_grad():
            trajectory = model.forward(initial_state.unsqueeze(0), time_points, external_inputs)
            trajectory = trajectory.squeeze(0).cpu().numpy()
        
        # Compute outputs
        outputs[i, 0] = np.trapz(trajectory[:, 0], dx=5/60)  # Glucose AUC
        outputs[i, 1] = np.max(trajectory[:, 1])  # Insulin peak
        outputs[i, 2] = np.mean(trajectory[6:, 3])  # GLP-1 response after meal
    
    # Perform Sobol analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for j, output_name in enumerate(output_names):
        Si = sobol.analyze(problem, outputs[:, j], print_to_console=False)
        
        # Plot first-order indices
        ax = axes[j]
        indices = Si['S1']
        ax.bar(param_names, indices)
        ax.set_xlabel('Parameters')
        ax.set_ylabel('First-order Sobol Index')
        ax.set_title(f'Sensitivity: {output_name}')
        ax.set_xticklabels(param_names, rotation=45)
        
        # Add values on bars
        for i, v in enumerate(indices):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure 2 saved to {save_path}")


def plot_fig3_posterior_bands(model: HybridODENN, test_loader, device: torch.device,
                             save_path: str = "results/figures/fig3_posterior_bands.png",
                             n_posterior_samples: int = 100):
    """
    Plot Figure 3: Posterior predictive bands showing uncertainty.
    
    Args:
        model: Trained model with variational parameters
        test_loader: Test data loader
        device: Computation device
        save_path: Path to save figure
        n_posterior_samples: Number of posterior samples
    """
    if not hasattr(model, 'variational_params') or model.variational_params is None:
        logger.warning("Model does not have variational parameters. Using point estimates.")
        plot_fig1_predictions(model, test_loader, device, save_path, n_subjects=1)
        return
    
    model.eval()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    state_names = ['Glucose (mmol/L)', 'Insulin (pmol/L)', 'GLP-1 (pmol/L)', 'Glucagon (pmol/L)']
    state_indices = [0, 1, 3, 2]
    
    # Get first batch
    batch = next(iter(test_loader))
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            for k, v in batch[key].items():
                batch[key][k] = v.to(device)
    
    # Get data for first subject
    initial_state = batch['initial_state'][0:1]
    targets = batch['observations'][0]
    time_points = batch['time_points'][0]
    external_inputs = {k: v[0:1] for k, v in batch.get('external_inputs', {}).items()}
    
    # Generate posterior samples
    logger.info("Generating posterior predictive samples...")
    predictions = []
    
    for i in range(n_posterior_samples):
        if i % 20 == 0:
            logger.info(f"  Sample {i}/{n_posterior_samples}")
        
        # Sample from posterior
        param_sample = model.variational_params.sample(1)[0]
        
        with torch.no_grad():
            pred = model.forward_with_params(
                param_sample, initial_state, time_points, external_inputs
            )
            predictions.append(pred.squeeze(0))
    
    # Stack predictions
    predictions = torch.stack(predictions)  # (n_samples, n_time, n_states)
    
    # Compute statistics
    pred_mean = predictions.mean(dim=0).cpu().numpy()
    pred_std = predictions.std(dim=0).cpu().numpy()
    pred_lower = np.percentile(predictions.cpu().numpy(), 2.5, axis=0)
    pred_upper = np.percentile(predictions.cpu().numpy(), 97.5, axis=0)
    
    time = time_points.cpu().numpy()
    
    # Plot each state
    for i, (state_idx, state_name) in enumerate(zip(state_indices, state_names)):
        ax = axes[i]
        
        # Plot ground truth
        ax.plot(time, targets[:, state_idx].cpu().numpy(),
               'k-', label='Ground Truth', linewidth=2)
        
        # Plot mean prediction
        ax.plot(time, pred_mean[:, state_idx],
               'r--', label='Mean Prediction', linewidth=2)
        
        # Plot uncertainty bands
        ax.fill_between(time, pred_lower[:, state_idx], pred_upper[:, state_idx],
                       color='red', alpha=0.3, label='95% Credible Interval')
        
        # Add one standard deviation band
        ax.fill_between(time, 
                       pred_mean[:, state_idx] - pred_std[:, state_idx],
                       pred_mean[:, state_idx] + pred_std[:, state_idx],
                       color='red', alpha=0.5, label='±1 SD')
        
        # Add meal indicators
        if external_inputs and 'meal' in external_inputs:
            meal_times = time[external_inputs['meal'][0].cpu().numpy() > 0]
            for meal_time in meal_times:
                ax.axvline(meal_time, color='green', alpha=0.3, linestyle=':', label='Meal' if meal_time == meal_times[0] else '')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(state_name)
        ax.set_title(f'{state_name} with Uncertainty')
        
        if i == 0:
            ax.legend(loc='best')
    
    plt.suptitle('Posterior Predictive Distribution', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure 3 saved to {save_path}")


def generate_performance_tables(results_dict: Dict[str, Dict[str, float]],
                               save_dir: str = "results/tables/"):
    """
    Generate Tables III-V showing performance metrics.
    
    Args:
        results_dict: Dictionary mapping experiment names to metrics
        save_dir: Directory to save tables
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Table III: Overall performance
    table3_data = []
    for exp_name, metrics in results_dict.items():
        row = {
            'Experiment': exp_name,
            'RMSE': f"{metrics.get('rmse', -1):.4f}",
            'MAE': f"{metrics.get('mae', -1):.4f}",
            'NRMSE': f"{metrics.get('nrmse', -1):.4f}"
        }
        table3_data.append(row)
    
    table3 = pd.DataFrame(table3_data)
    table3.to_csv(f"{save_dir}/table3_overall_performance.csv", index=False)
    table3.to_latex(f"{save_dir}/table3_overall_performance.tex", index=False)
    
    # Table IV: Per-state performance
    state_names = ['glucose', 'insulin', 'glucagon', 'glp1']
    table4_data = []
    
    for exp_name, metrics in results_dict.items():
        row = {'Experiment': exp_name}
        for state in state_names:
            rmse_key = f'rmse_{state}'
            if rmse_key in metrics:
                row[f'RMSE_{state.capitalize()}'] = f"{metrics[rmse_key]:.4f}"
        table4_data.append(row)
    
    table4 = pd.DataFrame(table4_data)
    table4.to_csv(f"{save_dir}/table4_per_state_performance.csv", index=False)
    table4.to_latex(f"{save_dir}/table4_per_state_performance.tex", index=False)
    
    # Table V: Ablation study results
    table5_data = []
    ablation_experiments = ['Full Model', 'No NN', 'No Bayes', 'No Physics']
    
    for exp_name in ablation_experiments:
        if exp_name in results_dict:
            metrics = results_dict[exp_name]
            row = {
                'Model': exp_name,
                'RMSE': f"{metrics.get('rmse', -1):.4f}",
                'MAE': f"{metrics.get('mae', -1):.4f}",
                'ECE': f"{metrics.get('ece', -1):.4f}" if 'ece' in metrics else 'N/A',
                'Coverage': f"{metrics.get('coverage_95', -1):.3f}" if 'coverage_95' in metrics else 'N/A'
            }
            table5_data.append(row)
    
    table5 = pd.DataFrame(table5_data)
    table5.to_csv(f"{save_dir}/table5_ablation_study.csv", index=False)
    table5.to_latex(f"{save_dir}/table5_ablation_study.tex", index=False)
    
    logger.info(f"Tables saved to {save_dir}")


def main():
    """
    Main function to generate all plots and tables.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all figures and tables')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/4gi_dataset.csv',
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results/',
                       help='Output directory for figures and tables')
    parser.add_argument('--figures', nargs='+', default=['all'],
                       choices=['all', 'fig1', 'fig2', 'fig3', 'tables'],
                       help='Which figures to generate')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint and create model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if args.config:
        from train.train_hybrid import load_config
        config = load_config(args.config)
    else:
        config = checkpoint.get('config', {})
    
    # Create model
    use_variational = config.get('model', {}).get('use_variational', False)
    model = HybridODENN(
        ode_params=None,
        nn_hidden=config.get('model', {}).get('nn_hidden', 64),
        nn_layers=config.get('model', {}).get('nn_layers', 4),
        use_variational=use_variational,
        device=device
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    config['data']['path'] = args.data
    from train.train_hybrid import create_data_loaders
    _, _, test_loader = create_data_loaders(config)
    
    # Generate figures
    if 'all' in args.figures or 'fig1' in args.figures:
        logger.info("Generating Figure 1...")
        plot_fig1_predictions(model, test_loader, device,
                            save_path=f"{args.output_dir}/figures/fig1_predictions.png")
    
    if 'all' in args.figures or 'fig2' in args.figures:
        logger.info("Generating Figure 2...")
        plot_fig2_sensitivity(model, device,
                            save_path=f"{args.output_dir}/figures/fig2_sensitivity.png")
    
    if 'all' in args.figures or 'fig3' in args.figures:
        logger.info("Generating Figure 3...")
        plot_fig3_posterior_bands(model, test_loader, device,
                                save_path=f"{args.output_dir}/figures/fig3_posterior_bands.png")
    
    if 'all' in args.figures or 'tables' in args.figures:
        logger.info("Generating performance tables...")
        
        # Evaluate model
        from eval.evaluate import evaluate_model
        metrics = evaluate_model(model, test_loader, device, use_variational)
        
        # Create results dictionary (you would have multiple experiments in practice)
        results_dict = {
            'Full Model': metrics,
            # Add more experiment results here
        }
        
        generate_performance_tables(results_dict, save_dir=f"{args.output_dir}/tables/")
    
    logger.info("All figures and tables generated successfully!")


if __name__ == "__main__":
    main()