"""Base model class for neural structural optimization."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_structural_optimization.problems_utils import ProblemParams
from .loss_structural import StructuralLoss
from .config import DEFAULT_MAX_ANALYSIS_DIM
from .utils import set_random_seed
from neural_structural_optimization import topo_api


class Model(nn.Module):
    """Base model class for structural optimization."""
    
    def __init__(
        self, 
        problem_params: Optional[ProblemParams | dict] = None, 
        clip_loss: Optional[object] = None, 
        seed: Optional[int] = None, 
        args: Optional[dict] = None
    ):
        super().__init__()
        
        # Handle problem parameters
        if problem_params is not None:
            if isinstance(problem_params, dict):
                self.problem_params = ProblemParams(**problem_params)
            else:
                self.problem_params = problem_params
                
            if args is None:
                problem = self.problem_params.get_problem()
                args = topo_api.specified_task(problem)
        else:
            self.problem_params = None

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

    def forward(self) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the design grid."""
        return (1, self.env.args['nely'], self.env.args['nelx'])

    def _get_mask_3d(self, H: int, W: int) -> torch.Tensor:
        """Get mask as 3D tensor with shape (1, H, W)."""
        mask = torch.as_tensor(
            self.args['mask'], 
            device=self.z.device, 
            dtype=self.z.dtype
        )
        
        if mask.ndim == 0:
            mask = torch.ones(
                (1, H, W), 
                dtype=self.z.dtype, 
                device=self.z.device
            ) * mask
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return mask

    def _update_problem_params(self, scale: Optional[float] = None) -> None:
        """Update problem parameters, typically for upsampling."""
        new_params = {}

        if scale is not None:
            new_params['width'] = int(self.problem_params.width * scale)
            new_params['height'] = int(self.problem_params.height * scale)
            new_params['rmin'] = self.problem_params.rmin * scale
            new_params['filter_width'] = self.problem_params.filter_width * scale
        
        self.problem_params = self.problem_params.copy(**new_params)
        problem = self.problem_params.get_problem()
        new_args = topo_api.specified_task(problem)
        self.env = topo_api.Environment(new_args)
        self.args = new_args
        
        # Update mask for new problem parameters
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
            'rmin': self.problem_params.rmin / f,
            'filter_width': self.problem_params.filter_width / f
        }

        analysis_params = self.problem_params.copy(**analysis_dict)
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

    def get_physics_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute physics-based structural loss."""
        if not hasattr(self, 'analysis_factor'):
            self._set_analysis_factor()
            
        if getattr(self, 'analysis_factor', 1) == 1:
            return StructuralLoss.apply(logits, self.env).mean()
            
        # Use downfactored structural grid
        z = self._downfactor_logits(logits)
        return StructuralLoss.apply(z, self.analysis_env).mean()
        
    def get_total_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute total loss (currently just physics loss)."""
        physics_loss = self.get_physics_loss(logits)
        return physics_loss
