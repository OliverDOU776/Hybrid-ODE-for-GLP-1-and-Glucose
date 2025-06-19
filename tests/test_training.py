"""
Test end-to-end training functionality.
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_ode_nn import HybridODENN
from train.train_hybrid import GlucoseDataset, train_epoch, validate
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


def create_test_dataset(n_subjects=2, n_timepoints=100):
    """Create a small test dataset."""
    # Create test data
    data = []
    
    for subject_id in range(n_subjects):
        time_hours = np.linspace(0, 5, n_timepoints)
        time_minutes = time_hours * 60
        
        # Simulate glucose dynamics
        glucose = 5.0 + 2.0 * np.sin(time_hours) + 0.5 * np.random.randn(n_timepoints)
        insulin = 100.0 + 50.0 * np.sin(time_hours + 0.5) + 10.0 * np.random.randn(n_timepoints)
        glucagon = 50.0 + 10.0 * np.sin(time_hours + 1.0) + 5.0 * np.random.randn(n_timepoints)
        glp1 = 20.0 + 10.0 * np.sin(time_hours + 1.5) + 2.0 * np.random.randn(n_timepoints)
        
        # Meal pattern
        meal_indicator = np.zeros(n_timepoints)
        meal_times = [30, 90, 150]  # Meals at 30, 90, 150 minutes
        for meal_time in meal_times:
            idx = int(meal_time / 300 * n_timepoints)
            if idx < n_timepoints:
                meal_indicator[idx] = 1.0
        
        # Create dataframe-like structure
        for i in range(n_timepoints):
            data.append({
                'subject_id': subject_id,
                'time_hours': time_hours[i],
                'time_minutes': time_minutes[i],
                'glucose_mmol_L': glucose[i],
                'insulin_pmol_L': insulin[i],
                'glucagon_pmol_L': glucagon[i],
                'glp1_pmol_L': glp1[i],
                'meal_indicator': meal_indicator[i]
            })
    
    return data


def test_dataset_creation():
    """Test dataset creation and data loading."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        import pandas as pd
        data = create_test_dataset(n_subjects=3, n_timepoints=100)
        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Create dataset
        dataset = GlucoseDataset(
            temp_file,
            sequence_length=20,
            stride=10,
            normalize=True
        )
        
        # Check dataset properties
        assert len(dataset) > 0, "Dataset is empty"
        assert len(dataset.state_cols) == 6, f"Expected 6 state columns, got {len(dataset.state_cols)}"
        
        # Check data item
        item = dataset[0]
        assert 'initial_state' in item
        assert 'observations' in item
        assert 'time_points' in item
        assert 'external_inputs' in item
        
        # Check shapes
        assert item['initial_state'].shape == (6,)
        assert item['observations'].shape == (20, 6)
        assert item['time_points'].shape == (20,)
        
    finally:
        # Clean up
        Path(temp_file).unlink()


def test_mini_training():
    """Test training for one mini-epoch."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        import pandas as pd
        data = create_test_dataset(n_subjects=2, n_timepoints=100)
        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dataset
        dataset = GlucoseDataset(temp_file, sequence_length=20, stride=10)
        
        # Use subset for faster testing
        subset_indices = list(range(min(10, len(dataset))))
        train_dataset = Subset(dataset, subset_indices)
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        
        # Create model
        device = torch.device('cpu')  # Use CPU for testing
        model = HybridODENN(
            nn_hidden=16,  # Smaller model for testing
            nn_layers=2,
            use_variational=False,
            device=device
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create config
        config = {
            'training': {
                'lambda1': 0.5,
                'lambda2': 0.1,
                'gradient_clip': 5.0
            },
            'ablation': {
                'no_physics': False
            }
        }
        
        # Create writer
        writer = SummaryWriter(temp_dir)
        
        # Train for one epoch
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        train_loss = train_epoch(
            model, train_loader, optimizer, None, 
            config, writer, epoch=0, device=device
        )
        
        # Check training occurred
        assert isinstance(train_loss, float), "Train loss should be float"
        assert train_loss > 0, "Train loss should be positive"
        assert not np.isnan(train_loss), "Train loss is NaN"
        
        # Check parameters updated
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                break
        assert params_changed, "Model parameters did not update"
        
        writer.close()
        
    finally:
        # Clean up
        Path(temp_file).unlink()
        shutil.rmtree(temp_dir)


def test_validation():
    """Test validation functionality."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        import pandas as pd
        data = create_test_dataset(n_subjects=2, n_timepoints=50)
        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Create dataset and loader
        dataset = GlucoseDataset(temp_file, sequence_length=20, stride=20)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create model
        device = torch.device('cpu')
        model = HybridODENN(
            nn_hidden=16,
            nn_layers=2,
            use_variational=False,
            device=device
        )
        
        # Create config
        config = {
            'training': {
                'lambda1': 0.5,
                'lambda2': 0.1
            },
            'ablation': {
                'no_physics': False
            }
        }
        
        # Run validation
        val_loss = validate(model, val_loader, config, device)
        
        # Check validation results
        assert isinstance(val_loss, float), "Val loss should be float"
        assert val_loss > 0, "Val loss should be positive"
        assert not np.isnan(val_loss), "Val loss is NaN"
        
    finally:
        # Clean up
        Path(temp_file).unlink()


def test_ablation_modes():
    """Test different ablation configurations."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create test batch
    batch = {
        'initial_state': torch.randn(2, 6),
        'observations': torch.randn(2, 10, 6),
        'time_points': torch.linspace(0, 1, 10),
        'external_inputs': {
            'meal': torch.zeros(2, 10),
            'tVNS': torch.zeros(2, 10)
        }
    }
    
    device = torch.device('cpu')
    
    # Test no NN ablation
    model_no_nn = HybridODENN(
        nn_hidden=16,
        nn_layers=2,
        use_variational=False,
        device=device
    )
    
    # Zero out NN parameters
    for param in model_no_nn.nn_residual.parameters():
        param.data.zero_()
        param.requires_grad = False
    
    loss_no_nn = model_no_nn.loss(batch, lambda1=1.0, lambda2=0.0)
    assert not torch.isnan(loss_no_nn), "Loss with no NN is NaN"
    
    # Test no physics ablation
    model_no_physics = HybridODENN(
        nn_hidden=16,
        nn_layers=2,
        use_variational=False,
        device=device
    )
    
    loss_no_physics = model_no_physics.loss(
        batch, lambda1=0.0, lambda2=0.1, use_physics_loss=False
    )
    assert not torch.isnan(loss_no_physics), "Loss with no physics is NaN"
    
    # Test no Bayes ablation (just L2 regularization)
    model_no_bayes = HybridODENN(
        nn_hidden=16,
        nn_layers=2,
        use_variational=False,
        device=device
    )
    
    loss_no_bayes = model_no_bayes.loss(batch, lambda1=1.0, lambda2=0.1)
    assert not torch.isnan(loss_no_bayes), "Loss with no Bayes is NaN"


def test_checkpoint_saving_loading():
    """Test model checkpoint save/load functionality."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model
        device = torch.device('cpu')
        model = HybridODENN(
            nn_hidden=32,
            nn_layers=3,
            use_variational=False,
            device=device
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Save checkpoint
        checkpoint_path = Path(temp_dir) / 'test_checkpoint.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'val_loss': 0.123,
            'config': {'test': True}
        }, checkpoint_path)
        
        # Create new model and load checkpoint
        model2 = HybridODENN(
            nn_hidden=32,
            nn_layers=3,
            use_variational=False,
            device=device
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Check parameters match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.allclose(param1, param2), f"Parameter value mismatch for {name1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])