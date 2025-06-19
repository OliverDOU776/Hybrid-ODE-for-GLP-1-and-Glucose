"""
Models package for hybrid ODE-NN framework.
"""

from .ode_core import ODECore
from .nn_residual import NNResidual
from .hybrid_ode_nn import HybridODENN
from .bayes import bayes_loss, VariationalParameters, compute_posterior_predictive

__all__ = [
    'ODECore',
    'NNResidual', 
    'HybridODENN',
    'bayes_loss',
    'VariationalParameters',
    'compute_posterior_predictive'
]