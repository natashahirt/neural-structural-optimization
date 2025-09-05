# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Structural Optimization Training Module.

This module provides a comprehensive training framework for neural structural optimization,
including various optimization algorithms and progressive training strategies.
"""

import warnings

# Import base classes
from .base import BaseOptimizer, OptimizationTracker

# Import optimizers
from .optimizers import (
    Adam_Optimizer,
    LBFGS_Optimizer,
    MMA_Optimizer,
    OptimalityCriteria_Optimizer,
)

# Import trainers
from .trainers import ProgressiveTrainer, PixelRefineTrainer

# Public API exports
__all__ = [
    # Base classes
    'BaseOptimizer',
    'OptimizationTracker',
    
    # Optimizers
    'Adam_Optimizer',
    'LBFGS_Optimizer', 
    'MMA_Optimizer',
    'OptimalityCriteria_Optimizer',
    
    # Trainers
    'ProgressiveTrainer',
    'PixelRefineTrainer',
    
    # Legacy functions (deprecated but still available)
    'train_adam',
    'train_lbfgs',
    'method_of_moving_asymptotes',
    'optimality_criteria',
    'train_progressive',
    'train_pixel_refine',
]
