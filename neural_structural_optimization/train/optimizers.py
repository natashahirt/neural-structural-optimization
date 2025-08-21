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

"""Optimization algorithms for neural structural optimization."""

from typing import Optional
import numpy as np
import torch
import xarray
from tqdm import tqdm

from neural_structural_optimization import models
from neural_structural_optimization.models import PixelModel, CNNModel

from .base import BaseOptimizer
from .utils import cosine_warmup, get_variables, constrained_logits, ensure_array_size


class Adam_Optimizer(BaseOptimizer):
    """Adam optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, lr_init: float = 1e-2, lr_final: float = 3e-3,
                 warmup_frac: float = 0.1, save_intermediate_designs: bool = True, 
                 grad_clip: Optional[float] = None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.warmup_frac = warmup_frac
        self.grad_clip = grad_clip
    
    def optimize(self) -> xarray.Dataset:
        """Run Adam optimization."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        
        for i in tqdm(range(self.max_iterations + 1), desc="Adam Optimizer"):
            lr = cosine_warmup(i, self.max_iterations, self.warmup_frac, self.lr_init, self.lr_final)
            
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = self.model()
            loss = self.model.get_total_loss(logits)
            
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            
            self.tracker.add_step(float(loss.detach()), logits.detach().cpu().numpy())
        
        return self.tracker.create_dataset(self.model)


class LBFGS_Optimizer(BaseOptimizer):
    """L-BFGS optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True,
                 lr: float = 1.0, history_size: int = 100, line_search: str = 'strong_wolfe',
                 tol_rel: float = 1e-3, tol_abs: float = 1e-2, patience: int = 5, 
                 min_steps: int = 10, coarse_start: bool = True):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.lr = lr
        self.history_size = history_size
        self.line_search = line_search
        self.tol_rel = tol_rel
        self.tol_abs = tol_abs
        self.patience = patience
        self.min_steps = min_steps
        self.coarse_start = coarse_start
    
    def optimize(self) -> xarray.Dataset:
        """Run L-BFGS optimization."""
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.lr,
            history_size=self.history_size,
            max_iter=1,
            line_search_fn=self.line_search
        )
        
        pbar = tqdm(range(self.max_iterations), desc="L-BFGS")
        prev_loss = None
        stall = 0
        fine_start = None
        fine_steps = 10
        
        for step in pbar:
            if not self.coarse_start:
                self.model.analysis_factor = 1
                self.model.analysis_env = self.model.env
            
            if isinstance(self.model, CNNModel) and step > 15:
                self.model._unfreeze_all()
            
            def closure():
                opt.zero_grad(set_to_none=True)
                logits = self.model()
                loss = self.model.get_total_loss(logits)
                loss.backward()
                return loss
            
            loss = opt.step(closure)
            loss_val = float(loss.detach())
            
            self.tracker.add_step(loss_val, self.model().detach().cpu().numpy())
            
            if prev_loss is not None:
                d = loss_val - prev_loss
                rel = abs(d) / (abs(prev_loss) + 1e-12)
                pbar.set_postfix({'loss': f'{loss_val:.6f}', 'Δ': f'{d:.2e}', 'relΔ': f'{rel:.2e}'})
                
                if (abs(d) <= self.tol_abs) or (rel <= self.tol_rel):
                    stall += 1
                else:
                    stall = 0
                
                if fine_start is None and (step + 1) >= self.min_steps and stall >= self.patience:
                    if (self.coarse_start and isinstance(self.model, PixelModel) and 
                        getattr(self.model, 'analysis_factor', 1) != 1):
                        self.model._set_analysis_factor(reset=True)
                        fine_start = step
                    else:
                        pbar.set_postfix({'loss': f'{loss_val:.6f}', 'stopped_at': step})
                        break
                
                if fine_start is not None and (step - fine_start + 1) >= fine_steps:
                    pbar.set_postfix({'loss': f'{loss_val:.6f}', 'stopped_at': step})
                    break
            
            prev_loss = loss_val
        
        return self.tracker.create_dataset(self.model)


class MMA_Optimizer(BaseOptimizer):
    """Method of Moving Asymptotes optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True, 
                 init_model=None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.init_model = init_model
        self._validate_model_type(models.PixelModel, "MMA")
    
    def optimize(self) -> xarray.Dataset:
        """Run MMA optimization."""
        import nlopt  # pylint: disable=g-import-not-at-top
        
        env = self.model.env
        if self.init_model is None:
            x0 = get_variables(self.model).astype(np.float64)
        else:
            x0 = constrained_logits(self.init_model).ravel()
        
        pbar = tqdm(total=self.max_iterations, desc="MMA Optimization")
        
        def objective(x):
            return env.objective(x, volume_constraint=False)
        
        def constraint(x):
            return env.constraint(x)
        
        def wrap_autograd_func(func, losses=None, frames=None):
            def wrapper(x, grad):
                if grad.size > 0:
                    value, grad[:] = autograd.value_and_grad(func)(x)
                else:
                    value = func(x)
                if losses is not None:
                    losses.append(value)
                if frames is not None:
                    frames.append(env.reshape(x).copy())
                    pbar.update(1)
                return value
            return wrapper
        
        opt = nlopt.opt(nlopt.LD_MMA, x0.size)
        opt.set_min_objective(wrap_autograd_func(objective, self.tracker.losses, self.tracker.frames))
        opt.add_inequality_constraint(wrap_autograd_func(constraint))
        opt.set_lower_bounds(1e-2)
        opt.set_upper_bounds(1.0)
        opt.set_maxeval(self.max_iterations)
        
        try:
            x = opt.optimize(x0)
            logging.info('MMA optimization completed successfully')
        except Exception as e:
            logging.info(f'MMA optimization stopped: {type(e).__name__}: {e}')
        finally:
            pbar.close()
        
        # Print min/max values across all designs
        designs_array = np.array(self.tracker.frames)
        print(f"Designs min value: {designs_array.min():.4f}")
        print(f"Designs max value: {designs_array.max():.4f}")
        
        return self.tracker.create_dataset(self.model)


class OptimalityCriteria_Optimizer(BaseOptimizer):
    """Optimality Criteria optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True, 
                 init_model=None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.init_model = init_model
        self._validate_model_type(models.PixelModel, "Optimality criteria")
    
    def optimize(self) -> xarray.Dataset:
        """Run Optimality Criteria optimization."""
        from neural_structural_optimization.structural import physics
        
        env = self.model.env
        nely, nelx = env.args['nely'], env.args['nelx']
        expected_size = nely * nelx
        
        # Initialize design
        if self.init_model is None:
            x = get_variables(self.model).astype(np.float64)
        else:
            x = constrained_logits(self.init_model).ravel()
        
        x = ensure_array_size(x, expected_size, "design array")
        
        for i in tqdm(range(self.max_iterations), desc="Optimality Criteria"):
            try:
                step_result = physics.optimality_criteria_step(x, env.ke, env.args)
                
                if isinstance(step_result, tuple):
                    x_new = step_result[1]
                else:
                    x_new = step_result
                
                x_new = ensure_array_size(x_new, expected_size, f"step {i} result")
                x = x_new
                
                loss = env.objective(x, volume_constraint=False)
                frame = x.reshape(nely, nelx)
                
                self.tracker.add_step(loss, frame.copy())
                
                if i % max(1, self.max_iterations // 10) == 0:
                    logging.info(f'step {i}, loss {loss:.6f}')
                    
            except Exception as e:
                logging.warning(f'Step {i} failed: {e}')
                break
        
        # Ensure we have results
        if not self.tracker.losses:
            self.tracker.losses = [0.0]
            self.tracker.frames = [np.zeros((nely, nelx))]
        
        return self.tracker.create_dataset(self.model)
