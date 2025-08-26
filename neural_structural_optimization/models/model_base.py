"""Base model class for neural structural optimization."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_structural_optimization.structural.problems import StructuralParams
from .loss_structural import StructuralLoss
from .loss_clip import CLIPLoss
from .config import DEFAULT_MAX_ANALYSIS_DIM
from .utils import set_random_seed
from neural_structural_optimization.structural import api as topo_api


class Model(nn.Module):
    """Base model class for structural optimization."""
    
    def __init__(
        self, 
        structural_params: Optional[StructuralParams | dict] = None, 
        clip_loss: Optional[object] = None, 
        seed: Optional[int] = None, 
        args: Optional[dict] = None
    ):
        super().__init__()
        
        # Handle problem parameters
        if structural_params is not None:
            if isinstance(structural_params, dict):
                self.structural_params = StructuralParams(**structural_params)
            else:
                self.structural_params = structural_params
                
            if args is None:
                problem = self.structural_params.get_problem()
                args = topo_api.specified_task(problem)
        else:
            self.structural_params = None

        # Set random seed
        set_random_seed(seed)
        self.seed = seed
        
        # Initialize environment
        self.env = topo_api.Environment(args)
        self.args = args
        
        # Initialize mask tensor once
        self.mask = torch.as_tensor(self.args['mask'], dtype=torch.float64)
        
        # Analysis settings
        self.analysis_factor = 1
        self.analysis_env = self.env

        if hasattr(self, 'z'):
            self.device = self.z.device
        else: 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        object.__setattr__(self, "clip_loss", None)  # placeholder
        if clip_loss is not None:
            clip_loss.clip_model = (
                clip_loss.clip_model.to(self.device).eval().requires_grad_(False)
            )
            if self.device.type == "cpu":
                clip_loss.clip_model = clip_loss.clip_model.float()
            if hasattr(clip_loss, "device"):
                clip_loss.device = self.device
            if 'clip_loss' in self._modules:
                del self._modules['clip_loss']
            object.__setattr__(self, "clip_loss", clip_loss)
        
    def forward(self) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the design grid."""
        return (1, self.env.args['nely'], self.env.args['nelx'])

    @property
    def class_name(self) -> str:
        """Get the name of the model class."""
        return self.__class__.__name__

    def _get_mask_3d(self, H: int, W: int) -> torch.Tensor:
        """Get mask as 3D tensor with shape (1, H, W)."""
        # Get device and dtype from z if available, otherwise use defaults
        if hasattr(self, 'z'):
            device = self.z.device
            dtype = self.z.dtype
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32
            
        mask = torch.as_tensor(
            self.args['mask'], 
            device=device, 
            dtype=dtype
        )
        
        if mask.ndim == 0:
            mask = torch.ones(
                (1, H, W), 
                dtype=dtype, 
                device=device
            ) * mask
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return mask

    def _update_structural_params(self, scale: Optional[float] = None) -> None:
        """Update structural parameters, typically for upsampling."""
        new_params = {}

        if scale is not None:
            new_params['width'] = int(self.structural_params.width * scale)
            new_params['height'] = int(self.structural_params.height * scale)
            new_params['rmin'] = self.structural_params.rmin * scale
            new_params['filter_width'] = self.structural_params.filter_width * scale
        
        self.structural_params = self.structural_params.copy(**new_params)
        problem = self.structural_params.get_problem()
        new_args = topo_api.specified_task(problem)
        self.env = topo_api.Environment(new_args)
        self.args = new_args
        
        # Update mask for new structural parameters
        self.mask = torch.as_tensor(self.args['mask'], dtype=torch.float64)

    def _set_analysis_factor(self, max_dim: int = DEFAULT_MAX_ANALYSIS_DIM, reset: bool = False) -> None:
        """Set analysis factor for downsampling during physics computation."""
        _, H, W = self.shape
        f = max(1, max((H + max_dim - 1) // max_dim, (W + max_dim - 1) // max_dim))
        self.analysis_factor = f

        if f == 1 or reset:
            self.analysis_env = self.env
            self.analysis_factor = 1
            return

        # Get analysis environment
        analysis_dict = {
            'width': int(round(W / f)),
            'height': int(round(H / f)),
            'rmin': self.structural_params.rmin / f,
            'filter_width': self.structural_params.filter_width / f
        }

        analysis_params = self.structural_params.copy(**analysis_dict)
        analysis_problem = analysis_params.get_problem()
        analysis_args = topo_api.specified_task(analysis_problem)
        analysis_args['volfrac'] = self.args.get('volfrac', analysis_args.get('volfrac', 0.5))

        self.analysis_env = topo_api.Environment(analysis_args)

    def _downfactor_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Downsample logits for analysis computation."""
        _, H, W = self.shape
        f = self.analysis_factor

        # Get "element densities" from analysis grid
        mask = self._get_mask_3d(H, W)

        z_4d = (z * mask).unsqueeze(0)  # (N=1, C=1, H, W)
        m_4d = mask.unsqueeze(0)

        num = F.avg_pool2d(z_4d, kernel_size=f, stride=f, ceil_mode=True)
        den = F.avg_pool2d(m_4d, kernel_size=f, stride=f, ceil_mode=True).clamp_min(1e-12)

        z_coarse = (num / den).squeeze(0) 
        return z_coarse

    def get_structural_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute physics-based structural loss."""
        if not hasattr(self, 'analysis_factor'):
            self._set_analysis_factor()
            
        if getattr(self, 'analysis_factor', 1) == 1:
            return StructuralLoss.apply(logits, self.env).mean()
            
        # Use downfactored structural grid
        z = self._downfactor_logits(logits)
        return StructuralLoss.apply(z, self.analysis_env).mean()

    def get_semantic_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute clip-based semantic loss."""
        if self.clip_loss is None:
            return logits.new_tensor(0.0)  # Return zero loss if no CLIP loss configured
        # Convert logits to images using sigmoid for CLIP loss
        return self.clip_loss(logits)
        
    def get_total_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute total loss (currently just physics loss)."""
        structural_loss = self.get_structural_loss(logits)
        semantic_loss = 0 #self.get_semantic_loss(logits)
        return structural_loss + semantic_loss
