"""
Main training script for the hybrid ODE-NN model.

Supports various configurations including:
- Different datasets (4GI, MIMIC)
- Solver choices (dopri5, rk45, etc.)
- Variational inference or point estimation
- Ablation studies (no NN, no Bayes)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import argparse
import yaml
import logging
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import json
from typing import Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_ode_nn import HybridODENN
from models.ode_core import ODECore
from inference.vi import VariationalInference
from inference.mcmc import run_nuts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlucoseDataset(torch.utils.data.Dataset):
    """
    Dataset for glucose-insulin dynamics.
    """
    
    def __init__(self, data_path: str, sequence_length: int = 61,
                 stride: int = 30, normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to CSV or Parquet file
            sequence_length: Length of each sequence
            stride: Stride between sequences
            normalize: Whether to normalize data
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            self.data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # State columns
        self.state_cols = ['glucose_mmol_L', 'insulin_pmol_L', 'glucagon_pmol_L', 
                          'glp1_pmol_L']
        
        # Add placeholder columns if missing
        if 'ge' not in self.data.columns:
            self.data['ge'] = 0.0
        if 'ffa' not in self.data.columns:
            self.data['ffa'] = 1.0
            
        self.state_cols.extend(['ge', 'ffa'])
        
        # External input columns
        self.input_cols = []
        if 'meal_indicator' in self.data.columns:
            self.input_cols.append('meal_indicator')
        if 'tvns' in self.data.columns:
            self.input_cols.append('tvns')
        else:
            self.data['tvns'] = 0.0
            self.input_cols.append('tvns')
        
        # Time column
        if 'time_minutes' in self.data.columns:
            self.data['time'] = self.data['time_minutes'] / 60.0  # Convert to hours
        elif 'time_hours' in self.data.columns:
            self.data['time'] = self.data['time_hours']
        else:
            # Assume 5-minute intervals
            self.data['time'] = np.arange(len(self.data)) * 5 / 60.0
        
        # Group by subject
        self.subject_groups = self.data.groupby('subject_id')
        self.subjects = list(self.subject_groups.groups.keys())
        
        # Create sequences
        self.sequences = []
        for subject_id in self.subjects:
            subject_data = self.subject_groups.get_group(subject_id)
            
            # Create sliding windows
            for start_idx in range(0, len(subject_data) - sequence_length + 1, stride):
                end_idx = start_idx + sequence_length
                sequence = subject_data.iloc[start_idx:end_idx]
                
                self.sequences.append({
                    'subject_id': subject_id,
                    'states': sequence[self.state_cols].values,
                    'inputs': sequence[self.input_cols].values,
                    'time': sequence['time'].values
                })
        
        # Compute normalization statistics
        if self.normalize:
            all_states = np.concatenate([seq['states'] for seq in self.sequences])
            self.state_mean = np.mean(all_states, axis=0)
            self.state_std = np.std(all_states, axis=0) + 1e-6
        else:
            self.state_mean = np.zeros(len(self.state_cols))
            self.state_std = np.ones(len(self.state_cols))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Normalize states
        states = (sequence['states'] - self.state_mean) / self.state_std
        
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        inputs_tensor = torch.tensor(sequence['inputs'], dtype=torch.float32)
        time_tensor = torch.tensor(sequence['time'], dtype=torch.float32)
        
        # Create batch item
        return {
            'initial_state': states_tensor[0],
            'observations': states_tensor,
            'time_points': time_tensor,
            'external_inputs': {
                'meal': inputs_tensor[:, 0] if 'meal_indicator' in self.input_cols else torch.zeros(len(time_tensor)),
                'tVNS': inputs_tensor[:, -1],  # tVNS is always last
            }
        }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    data_path = config['data']['path']
    dataset = GlucoseDataset(
        data_path,
        sequence_length=config['data'].get('sequence_length', 61),
        stride=config['data'].get('stride', 30),
        normalize=config['data'].get('normalize', True)
    )
    
    # Split dataset
    n_samples = len(dataset)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    logger.info(f"Dataset loaded: {n_samples} total samples")
    logger.info(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, scheduler, config, 
                writer, epoch, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    lambda1 = config['training'].get('lambda1', 1.0)
    lambda2 = config['training'].get('lambda2', 1.0)
    use_physics = not config['ablation'].get('no_physics', False)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for k, v in batch[key].items():
                    batch[key][k] = v.to(device)
        
        # Forward pass
        loss = model.loss(batch, lambda1=lambda1, lambda2=lambda2, 
                         use_physics_loss=use_physics)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log to TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, config, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    lambda1 = config['training'].get('lambda1', 1.0)
    lambda2 = config['training'].get('lambda2', 1.0)
    use_physics = not config['ablation'].get('no_physics', False)
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for k, v in batch[key].items():
                        batch[key][k] = v.to(device)
            
            # Compute loss
            loss = model.loss(batch, lambda1=lambda1, lambda2=lambda2,
                            use_physics_loss=use_physics)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train hybrid ODE-NN model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/4gi_dataset.csv',
                       help='Path to data file')
    parser.add_argument('--solver', type=str, default='dopri5',
                       choices=['dopri5', 'rk45', 'dop853', 'radau', 'bdf'],
                       help='ODE solver to use')
    parser.add_argument('--vi', action='store_true',
                       help='Use variational inference')
    parser.add_argument('--mcmc', action='store_true',
                       help='Use MCMC sampling')
    parser.add_argument('--no-nn', action='store_true',
                       help='Ablation: disable neural network')
    parser.add_argument('--no-bayes', action='store_true',
                       help='Ablation: disable Bayesian inference')
    parser.add_argument('--no-physics', action='store_true',
                       help='Ablation: disable physics loss')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'data': {
                'path': args.data,
                'sequence_length': 61,
                'stride': 30,
                'normalize': True,
                'num_workers': 0
            },
            'model': {
                'nn_hidden': 64,
                'nn_layers': 4,
                'solver': args.solver
            },
            'training': {
                'epochs': 300,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'lambda1': 1.0,
                'lambda2': 1.0 if not args.no_bayes else 0.0,
                'gradient_clip': 5.0,
                'early_stopping_patience': 20
            },
            'ablation': {
                'no_nn': args.no_nn,
                'no_bayes': args.no_bayes,
                'no_physics': args.no_physics
            }
        }
    
    # Override with command line arguments
    config['data']['path'] = args.data
    config['model']['solver'] = args.solver
    config['ablation']['no_nn'] = args.no_nn
    config['ablation']['no_bayes'] = args.no_bayes
    config['ablation']['no_physics'] = args.no_physics
    
    if args.no_bayes:
        config['training']['lambda2'] = 0.0
    
    # Create experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"hybrid_ode_nn_{timestamp}"
        if args.vi:
            experiment_name += "_vi"
        elif args.mcmc:
            experiment_name += "_mcmc"
        if args.no_nn:
            experiment_name += "_no_nn"
        if args.no_bayes:
            experiment_name += "_no_bayes"
    
    # Create directories
    log_dir = Path('runs') / experiment_name
    checkpoint_dir = Path('checkpoints') / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create model
    use_variational = args.vi and not args.no_bayes
    model = HybridODENN(
        ode_params=None,  # Use default parameters
        nn_hidden=config['model']['nn_hidden'],
        nn_layers=config['model']['nn_layers'],
        use_variational=use_variational,
        device=device
    ).to(device)
    
    # Disable NN if requested
    if args.no_nn:
        # Zero out NN parameters
        for param in model.nn_residual.parameters():
            param.data.zero_()
            param.requires_grad = False
    
    # Create optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if len(trainable_params) == 0:
        # If no trainable parameters (e.g., no-nn ablation), add a dummy parameter
        logger.warning("No trainable parameters found. Adding dummy parameter for optimizer.")
        dummy_param = torch.nn.Parameter(torch.zeros(1, device=device))
        model.register_parameter('_dummy_param', dummy_param)
        trainable_params = [dummy_param]
    
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config['training']['learning_rate']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    if args.vi and not args.no_bayes:
        # Variational inference training
        logger.warning("Variational inference training is currently under development. Using standard training instead.")
        logger.warning("To use Bayesian uncertainty, please use MCMC sampling after training.")
        
        # Fall back to standard training for now
        args.vi = False
        use_variational = False
        model.use_variational = False
        # Don't skip - continue with standard training below
    
    if not args.vi and args.mcmc and not args.no_bayes:
        # MCMC sampling (simplified - train MAP first)
        logger.info("Training MAP estimate before MCMC...")
        
        # Train for fewer epochs to get MAP
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(min(50, config['training']['epochs'])):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                   config, writer, epoch, device)
            val_loss = validate(model, val_loader, config, device)
            
            logger.info(f"Epoch {epoch}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, checkpoint_dir / 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                logger.info("Early stopping triggered")
                break
        
        # Run MCMC
        logger.info("Running MCMC sampling...")
        
        # Sample a batch for MCMC
        sample_batch = next(iter(val_loader))
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].to(device)
            elif isinstance(sample_batch[key], dict):
                for k, v in sample_batch[key].items():
                    sample_batch[key][k] = v.to(device)
        
        mcmc_samples = run_nuts(
            model, sample_batch,
            num_samples=1000,
            num_warmup=500,
            device=device
        )
        
        # Save MCMC results
        np.savez(checkpoint_dir / 'mcmc_samples.npz', **mcmc_samples)
        
    else:
        # Standard training
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['training']['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                   config, writer, epoch, device)
            val_loss = validate(model, val_loader, config, device)
            
            # Log to TensorBoard
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
            writer.add_scalar('val/epoch_loss', val_loss, epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_dir / 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': config
        }, checkpoint_dir / 'final_model.pth')
    
    writer.close()
    logger.info(f"Training completed. Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()