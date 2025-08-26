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

"""Progressive training strategies for neural structural optimization."""

from typing import Callable, List
import torch
import xarray

from neural_structural_optimization.models import PixelModel, CNNModel
from .utils import match_mean_std_in_logit_space


class ProgressiveTrainer:
    """Handles progressive training with upsampling."""
    
    def __init__(self, model, max_iterations: int, resize_num: int = 2, 
                 save_intermediate_designs: bool = True):
        self.model = model
        self.max_iterations = max_iterations
        self.resize_num = resize_num
        self.save_intermediate_designs = save_intermediate_designs
    
    def train(self, optimizer_class: Callable, **optimizer_kwargs) -> List[xarray.Dataset]:
        """Run progressive training."""
        ds_history = []
        
        for stage in range(self.resize_num):
            print(f"\nTraining stage {stage + 1}/{self.resize_num} at resolution: {self.model.shape[1]}x{self.model.shape[2]}")
            
            optimizer = optimizer_class(self.model, self.max_iterations, 
                                      save_intermediate_designs=self.save_intermediate_designs, 
                                      **optimizer_kwargs)
            ds = optimizer.optimize()
            ds_history.append(ds)
            
            if stage < self.resize_num - 1:
                self._upsample_model()
        
        return ds_history
    
    def _upsample_model(self):
        """Upsample the model for next stage."""
        if isinstance(self.model, PixelModel):
            self.model.upsample(scale=2, max_dim=500)
        elif isinstance(self.model, CNNModel):
            self.model.upsample(scale=2, freeze_transferred=True)

class PixelRefineTrainer(ProgressiveTrainer):
    """Handles progressive training with pixel refinement (CNN to PixelModel transition)."""
    
    def __init__(self, model, max_iterations: int, resize_num: int = 2, 
                 save_intermediate_designs: bool = True, switch_threshold: int = 200, coarse_start: bool = True):
        super().__init__(model, max_iterations, resize_num, save_intermediate_designs)
        self.switch_threshold = switch_threshold
        self.coarse_start = coarse_start
    
    def train(self, optimizer_class: Callable, **optimizer_kwargs) -> List[xarray.Dataset]:
        """Run progressive training with pixel refinement."""
        ds_history = []
        model = self.model
        
        for stage in range(self.resize_num):
            # Switch to PixelModel if CNN resolution exceeds threshold
            if isinstance(model, CNNModel) and max(model.shape[1], model.shape[2]) > self.switch_threshold:
                print(f"\nSwitching to PixelModel at resolution: {model.shape[1]}x{model.shape[2]}")
                pixel_model = PixelModel(
                    structural_params=model.structural_params,
                    clip_loss=model.clip_loss,
                    seed=model.seed
                )
                
                with torch.no_grad():
                    cnn_logits = model.forward()
                    pixel_model.z.data.copy_(cnn_logits)
                    
                    # Match statistics to ensure smooth transition
                    ref_img = torch.sigmoid(cnn_logits)
                    match_mean_std_in_logit_space(pixel_model.z, ref_img)
                
                model = pixel_model
                self.model = model  # Update the trainer's model reference

            print(f"\nTraining stage {stage + 1}/{self.resize_num} at resolution: {model.shape[1]}x{model.shape[2]}")
            print(f"Using coarse_start={self.coarse_start}")
            
            # Use the optimizer with the specified coarse_start setting
            optimizer_kwargs.setdefault('coarse_start', self.coarse_start)
            optimizer = optimizer_class(model, self.max_iterations, 
                                      save_intermediate_designs=self.save_intermediate_designs, 
                                      **optimizer_kwargs)
            ds = optimizer.optimize()
            ds_history.append(ds)
            
            if stage < self.resize_num - 1:
                self._upsample_model()
        
        return ds_history


class HybridTrainer(ProgressiveTrainer):
    """Handles progressive training with pixel refinement (CNN to PixelModel transition)."""
    
    def __init__(self, model, max_iterations: int, resize_num: int = 2, 
                 save_intermediate_designs: bool = True, switch_threshold: int = 200, coarse_start: bool = True):
        super().__init__(model, max_iterations, resize_num, save_intermediate_designs)
        self.switch_threshold = switch_threshold
        self.coarse_start = coarse_start
    
    def train(self, optimizer_class: Callable, **optimizer_kwargs) -> List[xarray.Dataset]:
        """Run progressive training with pixel refinement."""
        ds_history = []
        model = self.model
        
        for stage in range(self.resize_num):
            # Switch to PixelModel if CNN resolution exceeds threshold
            if isinstance(model, CNNModel) and max(model.shape[1], model.shape[2]) > self.switch_threshold:
                print(f"\nSwitching to PixelModel at resolution: {model.shape[1]}x{model.shape[2]}")
                pixel_model = PixelModel(
                    structural_params=model.structural_params,
                    clip_loss=model.clip_loss,
                    seed=model.seed
                )
                
                with torch.no_grad():
                    cnn_logits = model.forward()
                    pixel_model.z.data.copy_(cnn_logits)
                    
                    # Match statistics to ensure smooth transition
                    ref_img = torch.sigmoid(cnn_logits)
                    match_mean_std_in_logit_space(pixel_model.z, ref_img)
                
                model = pixel_model
                self.model = model  # Update the trainer's model reference

            print(f"\nTraining stage {stage + 1}/{self.resize_num} at resolution: {model.shape[1]}x{model.shape[2]}")
            print(f"Using coarse_start={self.coarse_start}")
            
            # Use the optimizer with the specified coarse_start setting
            optimizer_kwargs.setdefault('coarse_start', self.coarse_start)
            optimizer = optimizer_class(model, self.max_iterations, 
                                      save_intermediate_designs=self.save_intermediate_designs, 
                                      **optimizer_kwargs)
            ds = optimizer.optimize()
            ds_history.append(ds)
            
            if stage < self.resize_num - 1:
                self._upsample_model()
        
        return ds_history


