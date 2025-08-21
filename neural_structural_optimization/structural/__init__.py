"""Structural optimization module.

This module consolidates all structural optimization functionality including:
- Physics computation (topology optimization, finite element analysis)
- Problem definitions and utilities
- Autograd utilities for structural computations
- Loss functions for structural optimization
"""

from . import physics
from . import problems
from . import autograd
from . import api
from . import utils

__all__ = [
    'physics',
    'problems', 
    'autograd',
    'api',
    'utils'
]
