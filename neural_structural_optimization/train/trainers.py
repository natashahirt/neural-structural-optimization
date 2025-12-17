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
import torch.nn.functional as F
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
                 save_intermediate_designs: bool = True, switch_threshold: int = 200, coarse_start: bool = True,
                 initial_image: torch.Tensor = None):
        super().__init__(model, max_iterations, resize_num, save_intermediate_designs)
        self.switch_threshold = switch_threshold
        self.coarse_start = coarse_start
        self.initial_image = initial_image
    
    def _initialize_model_with_image(self, model):
        """Initialize model parameters with the provided image."""
        if self.initial_image is not None:
            print(f"Initializing model with provided image of shape: {self.initial_image.shape}")
            
            # Ensure image is in the right format and size
            if self.initial_image.dim() == 3:
                # Add batch dimension if needed
                img = self.initial_image.unsqueeze(0)
            else:
                img = self.initial_image
                
            # Resize image to match model's expected input size
            if img.shape[-2:] != (model.shape[1], model.shape[2]):
                img = F.interpolate(img, size=(model.shape[1], model.shape[2]), 
                                  mode='bilinear', align_corners=False)
            
            # Convert to logits (inverse sigmoid)
            with torch.no_grad():
                # Clamp to avoid log(0) or log(1)
                img_clamped = img.clamp(1e-6, 1.0 - 1e-6)
                logits = torch.logit(img_clamped)
                
                # Initialize model parameters
                if hasattr(model, 'z'):
                    # PixelModel
                    model.z.data.copy_(logits.squeeze(0))
                elif hasattr(model, 'parameters'):
                    # CNNModel - initialize first layer or use a custom initialization
                    # This is a simplified approach - you might need to adapt based on your model architecture
                    for param in model.parameters():
                        if param.dim() >= 2:  # Weight parameters
                            # Initialize with small random values around the image logits
                            param.data.normal_(0, 0.01)
                        else:  # Bias parameters
                            param.data.zero_()
                    
                    # Set the model's initial state to produce something close to the image
                    # This might require model-specific initialization logic
                    pass
    
    def train(self, optimizer_class: Callable, **optimizer_kwargs) -> List[xarray.Dataset]:
        """Run progressive training with pixel refinement."""
        ds_history = []
        model = self.model
        
        # Initialize model with image if provided
        if self.initial_image is not None:
            self._initialize_model_with_image(model)
        
        for stage in range(self.resize_num):
            if stage == 0:
                model.clip_loss.use_patch_pyramid = True
                model.clip_loss.global_downside = 100
                model.clip_loss.use_pairwise_spread = False
                model.clip_R = 7.0
            elif stage == 1:
                model.clip_loss.use_patch_pyramid = False
                model.clip_R = 2.0
            else:
                model.clip_loss.use_patch_pyramid = True
                if stage == 2:
                    model.clip_loss.crops_per_frac = (6, 10, 12)
                    model.clip_loss.min_patch_px = max(96, min(model.shape[1], model.shape[2]) // 4)
                    model.use_pairwise_spread = True
                    model.clip_R = 2.0
                if stage == 3:
                    model.clip_loss.patch_fracs = (0.75, 0.5, 0.25)
                    model.clip_loss.crops_per_frac = (8, 16, 24)

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
            
            if stage < self.resize_num - 1 and stage != 1:
                self._upsample_model()
        
        return ds_history
