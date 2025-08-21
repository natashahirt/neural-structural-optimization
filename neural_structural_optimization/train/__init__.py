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

# Legacy function wrappers for backward compatibility

def train_adam(model, max_iterations, lr_init=1e-2, lr_final=3e-3, warmup_frac=0.1,
               save_intermediate_designs=True, grad_clip=None):
    """Legacy wrapper for Adam optimization."""
    warnings.warn("train_adam is deprecated, use Adam_Optimizer instead", DeprecationWarning)
    return Adam_Optimizer(model, max_iterations, lr_init, lr_final, warmup_frac, 
                        save_intermediate_designs, grad_clip).optimize()

def train_lbfgs(model, max_iterations, save_intermediate_designs=True, lr=1.0, 
                history_size=100, line_search='strong_wolfe', tol_rel=1e-3, tol_abs=1e-2, 
                patience=5, min_steps=10, coarse_start=True):
    """Legacy wrapper for L-BFGS optimization."""
    warnings.warn("train_lbfgs is deprecated, use LBFGS_Optimizer instead", DeprecationWarning)
    return LBFGS_Optimizer(model, max_iterations, save_intermediate_designs, lr, history_size,
                         line_search, tol_rel, tol_abs, patience, min_steps, coarse_start).optimize()

def method_of_moving_asymptotes(model, max_iterations, save_intermediate_designs=True, init_model=None):
    """Legacy wrapper for MMA optimization."""
    warnings.warn("method_of_moving_asymptotes is deprecated, use MMA_Optimizer instead", DeprecationWarning)
    return MMA_Optimizer(model, max_iterations, save_intermediate_designs, init_model).optimize()

def optimality_criteria(model, max_iterations, save_intermediate_designs=True, init_model=None):
    """Legacy wrapper for Optimality Criteria optimization."""
    warnings.warn("optimality_criteria is deprecated, use OptimalityCriteria_Optimizer instead", DeprecationWarning)
    return OptimalityCriteria_Optimizer(model, max_iterations, save_intermediate_designs, init_model).optimize()

def train_progressive(model, max_iterations, resize_num=2, alg=train_adam, save_intermediate_designs=True):
    """Legacy wrapper for progressive training."""
    warnings.warn("train_progressive is deprecated, use ProgressiveTrainer instead", DeprecationWarning)
    return ProgressiveTrainer(model, max_iterations, resize_num, save_intermediate_designs).train(alg)

def train_pixel_refine(cnn_model, max_iterations, resize_num=2, alg=train_lbfgs, save_intermediate_designs=True):
    """Legacy wrapper for pixel refinement training."""
    warnings.warn("train_pixel_refine is deprecated, use PixelRefineTrainer instead", DeprecationWarning)
    
    # Convert legacy function to class
    if alg == train_lbfgs:
        optimizer_class = LBFGS_Optimizer
    elif alg == train_adam:
        optimizer_class = Adam_Optimizer
    else:
        # For other optimizers, we need to create a wrapper
        def optimizer_wrapper(model, max_iter, **kwargs):
            class WrapperOptimizer(BaseOptimizer):
                def __init__(self, model, max_iterations, **kwargs):
                    super().__init__(model, max_iterations, kwargs.get('save_intermediate_designs', True))
                    self.kwargs = kwargs
                
                def optimize(self):
                    return alg(self.model, self.max_iterations, **self.kwargs)
            
            return WrapperOptimizer(model, max_iter, **kwargs)
        
        optimizer_class = optimizer_wrapper
    
    trainer = PixelRefineTrainer(cnn_model, max_iterations, resize_num, save_intermediate_designs)
    return trainer.train(optimizer_class)

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
