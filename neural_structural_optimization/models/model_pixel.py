"""Pixel-based structural optimization model."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import Model
from .config import (
    DEFAULT_MAX_ANALYSIS_DIM,
    DEFAULT_TEMPERATURE,
    DEFAULT_EPSILON,
    DEFAULT_VOID_DENSITY,
    DEFAULT_PRESERVE_MEAN,
    DEFAULT_MAX_NEWTON_ITERATIONS,
    DEFAULT_MEAN_TOLERANCE,
    DEFAULT_STEP_TOLERANCE,
    DEFAULT_MEAN_CONVERGENCE,
)


class PixelModel(Model):
    """Direct pixel-wise optimization model."""
    
    def __init__(
        self, 
        problem_params: Optional[dict] = None, 
        clip_loss: Optional[object] = None, 
        seed: Optional[int] = None,
    ):
        super().__init__(problem_params=problem_params, clip_loss=clip_loss, seed=seed)
        
        # Initialize design parameters with volume fraction
        z_init = np.broadcast_to(
            self.env.args['volfrac'] * self.env.args['mask'], 
            self.shape
        )
        self.z = nn.Parameter(
            torch.tensor(z_init, dtype=torch.float64), 
            requires_grad=True
        )

    def forward(self) -> torch.Tensor:
        """Forward pass - return the design parameters directly."""
        return self.z

    def loss(self, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the total loss for the current design."""
        logits = self.forward()
        return self.get_total_loss(logits)

    @torch.no_grad()
    def upsample(
        self, 
        scale: int = 2, 
        preserve_mean: bool = DEFAULT_PRESERVE_MEAN, 
        max_dim: int = DEFAULT_MAX_ANALYSIS_DIM, 
        eps: float = DEFAULT_EPSILON, 
        void_density: float = DEFAULT_VOID_DENSITY
    ) -> None:
        """Upsample the design to higher resolution."""
    
        z_coarse = self.z.detach()
        _, H_coarse, W_coarse = self.shape

        # Get coarse mask and compute target mean
        mask_coarse = self._get_mask_3d(H_coarse, W_coarse)
        x_coarse = (z_coarse * mask_coarse).clamp(0.0, 1.0)
        target_mean = (x_coarse.sum() / mask_coarse.sum().clamp_min(1)).item()
        
        # Update problem parameters for new resolution
        self._update_problem_params(scale=scale)
        _, H_fine, W_fine = self.shape
        mask_fine = self._get_mask_3d(H_fine, W_fine)

        # Prepare coarse design for upsampling
        x_coarse_full = x_coarse + void_density * (1.0 - mask_coarse)
        z_log_coarse = torch.logit(x_coarse_full.clamp(eps, 1-eps))

        # Upsample using bilinear interpolation
        z_log_fine = F.interpolate(
            z_log_coarse.unsqueeze(0), 
            size=(H_fine, W_fine), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Handle void regions
        void_logit = torch.logit(
            torch.tensor(void_density, dtype=z_coarse.dtype, device=z_coarse.device).clamp(eps, 1-eps)
        )
        z_log_fine = torch.where(
            mask_fine > 0, 
            z_log_fine, 
            torch.full_like(z_log_fine, void_logit)
        )

        # Preserve mean density if requested
        b = 0.0
        if preserve_mean:
            # Try analytical first, fall back to Newton if needed
            current_mean = (torch.sigmoid(z_log_fine) * mask_fine).sum() / mask_fine.sum().clamp_min(1)
            
            if abs(current_mean - target_mean) > DEFAULT_MEAN_TOLERANCE:  # only if significant difference
                for _ in range(DEFAULT_MAX_NEWTON_ITERATIONS):  # max iterations
                    m = (torch.sigmoid(z_log_fine + b) * mask_fine).sum() / mask_fine.sum().clamp_min(1)
                    if abs(m - target_mean) < DEFAULT_MEAN_CONVERGENCE:
                        break
                    sigmoid_b = torch.sigmoid(z_log_fine + b)
                    dm_db = (sigmoid_b * (1 - sigmoid_b) * mask_fine).sum() / mask_fine.sum().clamp_min(1)
                    step = (target_mean - m) / (dm_db + 1e-8)
                    b += step
                    if abs(step) < DEFAULT_STEP_TOLERANCE:
                        break
        
        z_log_fine = z_log_fine + b
     
        # Apply temperature scaling and final processing
        z_fine = torch.sigmoid(z_log_fine / DEFAULT_TEMPERATURE)  # Apply temperature scaling
        z_fine = z_fine * mask_fine + void_density * (1.0 - mask_fine)

        # Update analysis settings and parameters
        self._set_analysis_factor(max_dim=max_dim)
        self.z = torch.nn.Parameter(z_fine, requires_grad=True)
