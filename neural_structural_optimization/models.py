import copy

import math
from typing import Iterable, Optional, Sequence, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint

from neural_structural_optimization.models_utils import (
    set_random_seed, batched_topo_loss, to_nchw, normalize_mask, AddOffset
)
from neural_structural_optimization.topo_physics import logit
from neural_structural_optimization.problems_utils import ProblemParams
from neural_structural_optimization import topo_api

# Optional: wire in your CLIP when ready
try:
    from neural_structural_optimization.clip_loss import CLIPLoss
except Exception:
    CLIPLoss = None


class StructuralLoss(torch.autograd.Function):
    """
    A bridge that lets PyTorch models optimize against your NumPy/HIPS-autograd
    physics. Forward detaches to NumPy; Backward pulls grads from autograd.
    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor, env):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")

        # store shape and device/dtype to rebuild grads 
        ctx.input_shape = logits.shape
        ctx.device = logits.device
        ctx.dtype = logits.dtype
        ctx.env = env

        # save detached tensor for backward 
        logits_cpu = logits.detach().cpu()  # keep original dtype; convert later
        ctx.save_for_backward(logits_cpu)

        x_np = logits.detach().cpu().double().numpy() # convert to double NumPy

        losses_np = batched_topo_loss(x_np, [env])  # -> shape (batch,)

        # return torch tensor
        return torch.as_tensor(np.asarray(losses_np), dtype=torch.float64, device=ctx.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        We re-run the NumPy/Autograd loss to get ∂loss/∂logits in NumPy,
        then map it back to Torch. grad_output is the upstream gradient
        on the (batch,) losses; we broadcast it onto the per-pixel grads.
        """
        (logits_cpu,) = ctx.saved_tensors
        env = ctx.env

        # NumPy double
        x_np = logits_cpu.double().numpy()

        # compute the gradient of sum(loss_i * grad_output_i) w.r.t. x
        # via HIPS autograd by defining a scalar objective.
        def scalar_objective(x_np):
            l = batched_topo_loss(x_np, [env]) # batched loss -> (batch,)
            go = grad_output.detach().cpu().to(torch.float64).numpy() # apply upstream grad (convert to numpy)
            return (l * go).sum()

        import autograd # only import autograd here so top-level file stays torch-only
        g_np = autograd.grad(scalar_objective)(x_np) # same shape as x_np

        # map back to torch, match original dtype & device
        g = torch.from_numpy(g_np).to(ctx.device).to(ctx.dtype)
        return g, None  # no grad for env
        

# ========================================
# Base model
# ========================================

class Model(nn.Module):
    def __init__(self, problem_params=None, clip_loss=None, seed=None, args=None):
        super().__init__()
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

        set_random_seed(seed)
        self.seed = seed
        self.env = topo_api.Environment(args)
        self.args = args
        
        self.analysis_factor = 1
        self.analysis_env = self.env

    def forward(self):
        # subclasses define self.z or a generator
        raise NotImplementedError

    # properties

    @property
    def shape(self):
        return (1, self.env.args['nely'], self.env.args['nelx'])

    # functions

    def _get_mask_3d(self, H, W):
        """Get mask as 3D tensor with shape (1, H, W)."""
        mask = torch.as_tensor(self.args['mask'], device=self.z.device, dtype=self.z.dtype)
        if mask.ndim == 0:
            mask = torch.ones((1, H, W), dtype=self.z.dtype, device=self.z.device) * mask
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask

    def _update_problem_params(self, scale=None):
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

    def _set_analysis_factor(self, max_dim: int = 500, reset=False):
        _, H, W = self.shape
        f = max(1, max((H + max_dim - 1) // max_dim, (W + max_dim - 1) // max_dim))
        self.analysis_factor = f

        if f == 1 or reset:
            self.analysis_env = self.env
            self.analysis_factor = 1
            return

        # get analysis environment
        analysis_dict = {
            'width'  : int(round(W / f)),
            'height' : int(round(H / f)),
            'rmin'   : self.problem_params.rmin / f,
            'filter_width' : self.problem_params.filter_width / f
        }

        analysis_params = self.problem_params.copy(**analysis_dict)
        analysis_problem = analysis_params.get_problem()
        analysis_args = topo_api.specified_task(analysis_problem)
        analysis_args['volfrac'] = self.args.get('volfrac', analysis_args.get('volfrac', 0.5))

        self.analysis_env = topo_api.Environment(analysis_args)

    def _downfactor_logits(self, z):
        _, H, W = self.shape
        f = self.analysis_factor

        # get "element densities" from analysis grid
        mask = self._get_mask_3d(H, W)

        z_4d = (z * mask).unsqueeze(0) # (N=1, C=1, H, W)
        m_4d = mask.unsqueeze(0)

        num = F.avg_pool2d(z_4d, kernel_size=f, stride=f, ceil_mode=True)
        den = F.avg_pool2d(m_4d, kernel_size=f, stride=f, ceil_mode=True).clamp_min(1e-12)

        z_coarse = (num / den).squeeze(0) 
        return z_coarse

    # losses

    def get_physics_loss(self, logits: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'analysis_factor'):
            self._set_analysis_factor()
            
        if getattr(self, 'analysis_factor', 1) == 1:
            return StructuralLoss.apply(logits, self.env).mean()
        # use downfactored structural grid
        z = self._downfactor_logits(logits)
        return StructuralLoss.apply(z, self.analysis_env).mean()
        
    def get_total_loss(self, logits: torch.Tensor) -> torch.Tensor:
        physics_loss = self.get_physics_loss(logits)
        return physics_loss


# ========================================
# Pixel model
# ========================================

class PixelModel(Model):
    def __init__(self, 
        problem_params=None, 
        clip_loss=None, 
        seed=None,
    ):
        super().__init__(problem_params=problem_params, clip_loss=clip_loss, seed=seed)
        z_init = np.broadcast_to(self.env.args['volfrac'] * self.env.args['mask'], self.shape)
        self.z = nn.Parameter(torch.tensor(z_init, dtype=torch.float64), requires_grad=True)

    def forward(self):
        return self.z

    def loss(self, logits=None):
        logits = self.forward()
        return self.get_total_loss(logits)

    def upsample(self, scale=2, preserve_mean=True, max_dim: int = 500):
        with torch.no_grad():
            z_coarse = self.z.detach()
            _, H_coarse, W_coarse = self.shape

            mask_coarse = self._get_mask_3d(H_coarse, W_coarse)
            
            # update problem        
            self._update_problem_params(scale=scale)

            _, H_fine, W_fine = self.shape
            z_fine = F.interpolate(z_coarse.unsqueeze(0), size=(H_fine, W_fine), mode='bilinear', align_corners=False).squeeze(0)
            z_fine = z_fine.clamp(0.0,1.0)

            if preserve_mean:
                mask_fine = self._get_mask_3d(H_fine, W_fine)
                mask_old = (z_coarse * mask_coarse).sum() / mask_coarse.sum().clamp_min(1e-12)
                mask_new = (z_fine * mask_fine).sum() / mask_fine.sum().clamp_min(1e-12)
                z_fine = (z_fine * (mask_old / mask_new.clamp_min(1e-12))).clamp(0.0, 1.0) * mask_fine
     
            # check for analysis problem
            self._set_analysis_factor(max_dim=max_dim)

        self.z = torch.nn.Parameter(z_fine, requires_grad=True)


# ========================================
# CNN
# ========================================

class IdentityNorm(nn.Module):
    def forward(self,x): return x

class AddOffset(nn.Module):
    """Per-channel learnable offset, scaled by `scale` (kept 0-init like TF AddOffset)."""
    def __init__(self, channels: int, scale: float = 10.0):
        super().__init__()
        self.scale = float(scale)
        self.offset = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W)
        return x + self.scale * self.offset.view(1, -1, 1, 1)

def _init_dense_orthogonal(dense: nn.Linear, latent_size: int, h_base: int, w_base: int, channels: int):
    fan_out = h_base * w_base * channels
    gain = math.sqrt(max(fan_out/latent_size, 1.0))
    nn.init.orthogonal_(dense.weight, gain=gain)
    if dense.bias is not None:
        nn.init.zeros_(dense.bias)

def _upsample_nn_to(x: torch.Tensor, size: Tuple[int,int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode='nearest')

class ConvStage(nn.Module):
    """
    One stage = activation -> UpSampling (nearest) -> normalization -> Conv2d -> AddOffset
    Mirrors the Keras block order.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | Tuple[int,int] = 5,
        norm_factor: Optional[Callable[[int], nn.Module]] = None,
        offset_scale: float = 10.0,
        neg_slope: float = 0.2,
        conv_initializer: str = "kaiming_fan_in",  # mirrors tf.VarianceScaling(fan_in)
        stage_name: str = "",
    ):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        self.norm = (norm_factor(in_ch) if norm_factor is not None else IdentityNorm())
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding="same", bias=True)
        self.offset = AddOffset(out_ch, scale=offset_scale)
        # init conv
        if conv_initializer == "kaiming_fan_in":
            nn.init.kaiming_normal_(self.conv.weight, a=neg_slope, mode="fan_in", nonlinearity="leaky_relu")
        elif conv_initializer == "kaiming_fan_out":
            nn.init.kaiming_normal_(self.conv.weight, a=neg_slope, mode="fan_out", nonlinearity="leaky_relu")
        else:
            raise ValueError(f"Unknown conv_initializer: {conv_initializer}")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.stage_name = stage_name  # used by the adaptive transfer logic

    def forward(self, x: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
        x = self.act(x)
        x = _upsample_nn_to(x, target_hw)
        x = self.norm(x)
        x = self.conv(x)
        x = self.offset(x)
        return x


class CNNModel(Model):
    def __init__(
        self,
        problem_params=None,
        latent_size: int = 128,
        dense_channels: int = 32,
        conv_upsample: Sequence[int] = (1, 2, 2, 2, 1),
        conv_filters: Sequence[int] = (128, 64, 32, 16, 1),
        offset_scale: float = 10.0,
        kernel_size: int | Tuple[int,int] = 5,
        norm: Optional[Callable[[int], nn.Module]] = lambda C: nn.InstanceNorm2d(num_features=int(C), affine=False, track_running_stats=False),
        conv_initializer: str = "kaiming_fan_in",
        neg_slope: float = 0.2,
        seed: Optional[int] = None,
    ):
        super().__init__(problem_params=problem_params, seed=seed)

        if len(conv_upsample) != len(conv_filters):
            raise ValueError("conv_upsample and conv_filters must have same length")
        if conv_filters[-1] != 1:
            raise ValueError("conv_filters[-1] must be 1 to produce (1,H,W) logits")
    
        self.problem_params = problem_params
        self.latent_size = int(latent_size)
        self.dense_channels = int(dense_channels)
        self.conv_upsample = conv_upsample
        self.conv_filters = conv_filters
        self.offset_scale = offset_scale
        self.kernel_size = kernel_size
        self.norm = norm
        self.conv_initializer = conv_initializer
        self.neg_slope = neg_slope

        if seed is not None:
            g = torch.Generator(device='cpu')
            g.manual_seed(int(seed))
            z_init = torch.randn(1, self.latent_size, generator=g) 
        else:
            z_init = torch.randn(1, self.latent_size)

        self.z = nn.Parameter(z_init)

        self._compute_base_sizes()
        self._build()


        self.register_buffer("_distill_target", None, persistent=False)
        self.register_buffer("_rho_hi_logit", None, False)
        self._distill_scale = 1
        self._distill_weight = 0.0
        self._warm_tau = 0.0

    @property
    def shape(self):
        return (1, int(self.env.args['nely']), int(self.env.args['nelx']))

    def _compute_base_sizes(self):
        total_up = 1
        for u in self.conv_upsample:
            total_up *= int(u)

        _, H, W = self.shape

        if H % total_up != 0 or W % total_up != 0:
            raise ValueError(f"Grid {H}x{W} too small,not divisible by Π(conv_upsample)={total_up}")

        self.h_base = H // total_up
        self.w_base = W // total_up

        sizes = []
        ch = self.dense_channels
        h, w, = self.h_base, self.w_base
        for u in self.conv_upsample:
            h *= int(u); w *= int(u)
            sizes.append((h, w))
        self.stage_sizes = sizes

    def _build(self): 
        self.dense  = nn.Linear(self.latent_size, self.h_base * self.w_base * self.dense_channels)
        _init_dense_orthogonal(self.dense, self.latent_size, self.h_base, self.w_base, self.dense_channels)        

        # convs + offsets
        stages = []
        in_ch = self.dense_channels
        for i, (out_ch, target_hw) in enumerate(zip(self.conv_filters, self.stage_sizes)):
            stage_name = f"stage_{i:02d}_C{in_ch}->{out_ch}"
            stage = ConvStage(
                in_ch, out_ch,
                kernel_size=self.kernel_size,
                norm_factor=self.norm,
                offset_scale=self.offset_scale,
                neg_slope=self.neg_slope,
                conv_initializer=self.conv_initializer,
                stage_name=stage_name,
            )
            stages.append(stage)
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self) -> torch.Tensor:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            z = self.z.float()
            x = self.dense(z).view(1, self.dense_channels, self.h_base, self.w_base).contiguous(memory_format=torch.channels_last)

            # upsample conv stages
            for stage, target_hw in zip(self.stages, self.stage_sizes):
                x = stage(x, target_hw)

            if self._warm_tau > 0.0 and (self._rho_hi_logit is not None) and (self._rho_hi_logit.shape[-2:] == x.shape[-2:]):
                x = (1.0 - self._warm_tau) * x + self._warm_tau * self._rho_hi_logit[None, None]

        # (1,1,H,W) -> (1,H,W) as float64 for physics
        return x[:, 0, :, :].to(torch.float64)

    def loss(self, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits is None:
            logits = self.forward()

        total = self.get_total_loss(logits)

        if self._distill_weight > 0.0 and self._distill_target is not None and self._distill_scale >= 1:
            x = logits.to(torch.float32)
            if x.ndim == 3:
                x = x[None, None]
            if self._distill_scale > 1:
                x = F.avg_pool2d(x, kernel_size=self._distill_scale, stride=self._distill_scale)
            student_ds = x[0,0]
            distill = F.mse_loss(student_ds, self._distill_target)
            total = total + float(self._distill_weight) * distill

        return total

    @torch.no_grad()
    def upsample(self, scale:int, *, freeze_transferred=True, distill_weight:float=0.1):
    
        # teacher at old res -> nearest upsample to new res -> logit target
        old_logits = self.forward()[0].to(torch.float32).contiguous() # (H_old, W_old)
        rho_lo = torch.sigmoid(old_logits)
        rho_hi = F.interpolate(rho_lo[None, None], scale_factor=scale, mode="nearest")[0,0]
        rho_hi = rho_hi.clamp_(0.01, 0.99)
        rho_hi_logit = logit(rho_hi)

        self._update_problem_params(scale=scale)
        ups = list(map(int, self.conv_upsample))
        ups[-1] *= scale

        self._rebuild(tuple(ups), rho_hi_logit, old_logits, scale, distill_weight, freeze_transferred)

    def _rebuild(self, ups, rho_hi_logit, distill_target_logits, scale, distill_weight, freeze_transferred):
        
        prev_state_dict = self.state_dict()
        self.conv_upsample = tuple(int(u) for u in ups)
        self._compute_base_sizes()
        old_keys = set(prev_state_dict.keys())

        new_self = CNNModel(
            problem_params=self.problem_params,
            latent_size=self.latent_size,
            dense_channels=self.dense_channels,
            conv_upsample=self.conv_upsample,
            conv_filters=self.conv_filters,
            offset_scale=self.offset_scale,
            kernel_size=self.kernel_size,
            norm=self.norm,
            conv_initializer=self.conv_initializer,
            neg_slope=self.neg_slope,
        )

        new_state_dict = new_self.state_dict()

        copy_map = {}
        for k,v in new_state_dict.items():
            if k in old_keys and prev_state_dict[k].shape == v.shape:
                copy_map[k] = prev_state_dict[k].clone()

        new_state_dict.update(copy_map)
        self.load_state_dict(new_state_dict)
        
        def should_freeze(name: str) -> bool:
            if name == "z" or name.startswith(f"stages.{len(self.stages) - 1}."):  # keep last stage plastic
                return False
            return name in copy_map  # only freeze transferred early layers

        if freeze_transferred:
            for name, p in self.named_parameters():
                p.requires_grad_(not should_freeze(name))

        if not any(p.requires_grad for p in self.parameters()):
            self.z.requires_grad_(True)
        
        self.register_buffer("_rho_hi_logit", rho_hi_logit, persistent=False)
        self.register_buffer("_distill_target", distill_target_logits.detach().to(torch.float32), persistent=False)
        self._distill_scale = int(scale)
        self._distill_weight = float(distill_weight)
        self._warm_tau = 0.25

    def _unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self._warm_tau = 0.0