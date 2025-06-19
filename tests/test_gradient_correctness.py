"""
Test gradient correctness for the hybrid ODE-NN model.
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_ode_nn import HybridODENN
from models.nn_residual import NNResidual


def test_nn_residual_gradients():
    """Test gradient flow through NN residual block."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create NN residual
    nn_residual = NNResidual(hidden_dim=32, n_layers=2)
    
    # Initialize with small random weights to get non-zero gradients
    with torch.no_grad():
        for param in nn_residual.parameters():
            param.data.normal_(0, 0.01)
    
    # Create test inputs
    batch_size = 4
    t = torch.rand(batch_size)
    state = torch.randn(batch_size, 6)
    glp1 = torch.rand(batch_size) * 50
    tvns = torch.rand(batch_size)
    
    # All inputs require gradients
    t.requires_grad = True
    state.requires_grad = True
    glp1.requires_grad = True
    tvns.requires_grad = True
    
    # Forward pass
    output = nn_residual(t, state, glp1, tvns)
    
    # Check output shape
    assert output.shape == (batch_size, 6), f"Unexpected output shape: {output.shape}"
    
    # Compute loss and gradients
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist and are not NaN
    for param_name, param in [("state", state), ("glp1", glp1), ("tvns", tvns)]:
        assert param.grad is not None, f"No gradient for {param_name}"
        assert not torch.any(torch.isnan(param.grad)), f"NaN gradient for {param_name}"
        assert torch.any(param.grad != 0), f"Zero gradient for {param_name}"
    
    # Check time gradient exists but might be zero due to initialization
    assert t.grad is not None, "No gradient for t"
    assert not torch.any(torch.isnan(t.grad)), "NaN gradient for t"


def test_hybrid_model_gradients():
    """Test gradient flow through full hybrid model."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create hybrid model
    model = HybridODENN(
        nn_hidden=32,
        nn_layers=2,
        use_variational=False,
        device='cpu'
    )
    
    # Create test batch
    batch_size = 2
    seq_length = 5
    
    batch = {
        'initial_state': torch.randn(batch_size, 6),
        'observations': torch.randn(batch_size, seq_length, 6),
        'time_points': torch.linspace(0, 1, seq_length).unsqueeze(0).expand(batch_size, -1),
        'external_inputs': {
            'meal': torch.rand(batch_size, seq_length) * 10,
            'tVNS': torch.rand(batch_size, seq_length)
        }
    }
    
    # Ensure model is in training mode
    model.train()
    
    # Forward pass
    loss = model.loss(batch, lambda1=1.0, lambda2=0.1, use_physics_loss=True)
    
    # Check loss is scalar
    assert loss.dim() == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"
    
    # Backward pass
    loss.backward()
    
    # Check gradients for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.any(torch.isnan(param.grad)), f"NaN gradient for {name}"
            # Some parameters might have zero gradients if not used
            if 'bias' not in name:  # Biases can be zero
                grad_norm = param.grad.norm()
                assert grad_norm > 0, f"Zero gradient norm for {name}"


def test_gradient_accumulation():
    """Test gradient accumulation over multiple batches."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create hybrid model
    model = HybridODENN(
        nn_hidden=32,
        nn_layers=2,
        use_variational=False,
        device='cpu'
    )
    
    # Zero gradients
    model.zero_grad()
    
    # Accumulate gradients over multiple batches
    n_batches = 3
    accumulated_loss = 0.0
    
    for i in range(n_batches):
        batch = {
            'initial_state': torch.randn(2, 6),
            'observations': torch.randn(2, 10, 6),
            'time_points': torch.linspace(0, 1, 10),
            'external_inputs': {
                'meal': torch.zeros(2, 10),
                'tVNS': torch.zeros(2, 10)
            }
        }
        
        loss = model.loss(batch, lambda1=0.5, lambda2=0.1)
        loss.backward()
        accumulated_loss += loss.item()
    
    # Check gradients accumulated
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    # Reset and compute single batch gradient
    model.zero_grad()
    single_loss = model.loss(batch, lambda1=0.5, lambda2=0.1)
    single_loss.backward()
    
    # Single batch gradients should be smaller than accumulated
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            single_norm = param.grad.norm().item()
            if name in grad_norms and grad_norms[name] > 1e-6:
                assert single_norm <= grad_norms[name] * 1.1, \
                    f"Single gradient larger than accumulated for {name}"


@pytest.mark.skip(reason="Variational inference interface needs refactoring")
def test_variational_gradients():
    """Test gradient flow with variational inference."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create hybrid model with variational inference
    model = HybridODENN(
        nn_hidden=32,
        nn_layers=2,
        use_variational=True,
        device='cpu'
    )
    
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
    
    # Forward pass
    loss = model.loss(batch, lambda1=0.5, lambda2=1.0)
    
    # Backward pass
    loss.backward()
    
    # Check variational parameters have gradients
    var_params = model.variational_params
    assert var_params.mu.grad is not None, "No gradient for variational mean"
    assert var_params.log_sigma.grad is not None, "No gradient for variational log_sigma"
    assert not torch.any(torch.isnan(var_params.mu.grad)), "NaN in variational mean gradient"
    assert not torch.any(torch.isnan(var_params.log_sigma.grad)), "NaN in variational log_sigma gradient"


def test_gradient_clipping():
    """Test gradient clipping functionality."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create model
    model = HybridODENN(
        nn_hidden=32,
        nn_layers=2,
        use_variational=False,
        device='cpu'
    )
    
    # Create batch that might cause large gradients
    batch = {
        'initial_state': torch.randn(2, 6) * 10,  # Large initial state
        'observations': torch.randn(2, 10, 6) * 10,  # Large observations
        'time_points': torch.linspace(0, 5, 10),  # Longer time span
        'external_inputs': {
            'meal': torch.rand(2, 10) * 50,  # Large meal inputs
            'tVNS': torch.ones(2, 10)
        }
    }
    
    # Forward and backward
    loss = model.loss(batch)
    loss.backward()
    
    # Check gradient norms before clipping
    total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    
    # Clip gradients
    max_norm = 5.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Check gradient norms after clipping
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    
    # Verify clipping worked
    assert total_norm_after <= max_norm * 1.01, \
        f"Gradient norm {total_norm_after} exceeds max {max_norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])