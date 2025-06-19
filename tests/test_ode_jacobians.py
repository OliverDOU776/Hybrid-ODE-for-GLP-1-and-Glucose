"""
Test ODE Jacobian computations using finite differences and autograd.
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ode_core import ODECore


def finite_difference_jacobian(func, x, eps=1e-6):
    """
    Compute Jacobian using finite differences.
    
    Args:
        func: Function to differentiate
        x: Input tensor
        eps: Finite difference step size
        
    Returns:
        jacobian: Jacobian matrix
    """
    x = x.clone().detach().requires_grad_(False)
    batch_size = x.shape[0] if x.dim() > 1 else 1
    n_dims = x.shape[-1]
    
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    # Evaluate function at x
    f0 = func(x).detach()
    n_outputs = f0.shape[-1]
    
    # Initialize Jacobian
    jacobian = torch.zeros(batch_size, n_outputs, n_dims)
    
    # Compute finite differences
    for i in range(n_dims):
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus[:, i] += eps
        x_minus[:, i] -= eps
        
        f_plus = func(x_plus).detach()
        f_minus = func(x_minus).detach()
        
        jacobian[:, :, i] = (f_plus - f_minus) / (2 * eps)
    
    return jacobian.squeeze(0) if batch_size == 1 else jacobian


def test_ode_jacobian_basic():
    """Test ODE Jacobian computation for basic case."""
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create ODE core
    ode_core = ODECore()
    
    # Create test state
    state = torch.tensor([
        5.0,    # glucose
        100.0,  # insulin
        50.0,   # glucagon
        20.0,   # GLP-1
        0.0,    # gastric emptying
        1.0     # FFA
    ], dtype=torch.float32, requires_grad=True)
    
    t = torch.tensor(0.0)
    external_inputs = {
        'meal': torch.tensor(0.0),
        'tVNS': torch.tensor(0.0)
    }
    
    # Define function for Jacobian computation
    def ode_func(s):
        return ode_core(t, s, external_inputs)
    
    # Compute Jacobian using autograd
    output = ode_func(state)
    jacobian_auto = torch.autograd.functional.jacobian(ode_func, state)
    
    # Compute Jacobian using finite differences
    jacobian_fd = finite_difference_jacobian(ode_func, state)
    
    # Compare with relaxed tolerance due to numerical precision in finite differences
    max_diff = (jacobian_auto - jacobian_fd).abs().max()
    assert torch.allclose(jacobian_auto, jacobian_fd, rtol=0.1, atol=0.1), \
        f"Jacobian mismatch: max diff = {max_diff}"


def test_ode_jacobian_batch():
    """Test ODE Jacobian computation with batched inputs."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create ODE core
    ode_core = ODECore()
    
    # Create batched test states
    batch_size = 4
    states = torch.randn(batch_size, 6) * 0.1 + torch.tensor([
        [5.0, 100.0, 50.0, 20.0, 0.0, 1.0]
    ])
    states.requires_grad = True
    
    t = torch.zeros(batch_size)
    external_inputs = {
        'meal': torch.zeros(batch_size),
        'tVNS': torch.zeros(batch_size)
    }
    
    # Compute derivatives
    output = ode_core(t, states, external_inputs)
    
    # Check output shape
    assert output.shape == (batch_size, 6), f"Unexpected output shape: {output.shape}"
    
    # Check that gradients flow
    loss = output.sum()
    loss.backward()
    assert states.grad is not None, "Gradients not computed"
    assert not torch.any(torch.isnan(states.grad)), "NaN gradients detected"


def test_ode_jacobian_with_inputs():
    """Test ODE Jacobian with external inputs."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create ODE core
    ode_core = ODECore()
    
    # Create test state
    state = torch.tensor([
        8.0,    # glucose (elevated)
        150.0,  # insulin (elevated)
        40.0,   # glucagon
        30.0,   # GLP-1 (elevated)
        0.5,    # gastric emptying
        1.2     # FFA
    ], dtype=torch.float32, requires_grad=True)
    
    t = torch.tensor(1.0)
    
    # Test with meal input
    external_inputs = {
        'meal': torch.tensor(10.0),  # 10 g carbohydrate
        'tVNS': torch.tensor(0.0)
    }
    
    output_meal = ode_core(t, state, external_inputs)
    
    # Test with tVNS input
    external_inputs['meal'] = torch.tensor(0.0)
    external_inputs['tVNS'] = torch.tensor(1.0)  # tVNS active
    
    output_tvns = ode_core(t, state, external_inputs)
    
    # Outputs should be different
    assert not torch.allclose(output_meal, output_tvns), \
        "ODE outputs should differ with different inputs"


def test_ode_stability():
    """Test ODE numerical stability."""
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create ODE core
    ode_core = ODECore()
    
    # Test with extreme values
    extreme_states = [
        torch.tensor([20.0, 500.0, 200.0, 100.0, 2.0, 5.0]),  # Very high
        torch.tensor([2.0, 10.0, 10.0, 5.0, 0.0, 0.1]),       # Very low
        torch.tensor([5.0, 100.0, 50.0, 20.0, 0.0, 1.0]),     # Normal
    ]
    
    t = torch.tensor(0.0)
    external_inputs = {
        'meal': torch.tensor(0.0),
        'tVNS': torch.tensor(0.0)
    }
    
    for i, state in enumerate(extreme_states):
        state.requires_grad = True
        output = ode_core(t, state, external_inputs)
        
        # Check for NaN or Inf
        assert not torch.any(torch.isnan(output)), f"NaN detected in output for state {i}"
        assert not torch.any(torch.isinf(output)), f"Inf detected in output for state {i}"
        
        # Check gradients
        loss = output.sum()
        loss.backward()
        assert not torch.any(torch.isnan(state.grad)), f"NaN gradients for state {i}"
        assert not torch.any(torch.isinf(state.grad)), f"Inf gradients for state {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])