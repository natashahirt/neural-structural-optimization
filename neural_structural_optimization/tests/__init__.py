"""Test suite for neural structural optimization."""

# Import test modules for easy access
from . import test_autograd
from . import test_physics
from . import test_pipeline

__all__ = [
    'test_autograd',
    'test_physics', 
    'test_pipeline'
]
