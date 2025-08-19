import copy

import math
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

def _init_dense_orthogonal(dense, latent_size, h_base, w_base, channels):
    fan_out = h_base * w_base * channels
    gain = math.sqrt(max(fan_out / latent_size, 1.0))
    nn.init.orthogonal_(dense.weight, gain=gain)
    if dense.bias is not None:
        nn.init.zeros_(dense.bias)

def _upsample_to(x, size, mode='bilinear'):
    """Resize NCHW tensor x to (H,W)=size with TF/Keras-parity semantics."""
    if x.shape[-2:] == size:
        return x
    if mode == 'nearest':
        return F.interpolate(x, size=size, mode='nearest')
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, activation=nn.LeakyReLU):
        super().__init__()
        layers_ = [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding="same", bias=True)]
        self.conv_block = nn.Sequential(*layers_)
    
    def forward(self, x): return self.conv_block(x)

class CNNModel(Model):
    def __init__(
        self,
        problem_params=None,
        latent_size=128,
        init_channels=32,
        conv_upsample=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16),
        kernel_size=5,
        offset_scale=10,
        activation=nn.LeakyReLU,  
        norm=None, # nn.BatchNorm2d or lambda C: nn.GroupNorm(8, C)
        final_activation=None,
        seed=None,
    ):
        super().__init__(problem_params=problem_params, seed=seed)

        if len(conv_upsample) != len(conv_filters):
            raise ValueError("conv_upsample and conv_filters must have same length")

        self.problem_params = problem_params
        self.latent_size = int(latent_size)
        self.init_channels = int(init_channels)
        self.conv_upsample = conv_upsample
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.offset_scale = float(offset_scale)
        self.norm = norm
        self.final_activation = final_activation
        self.seed = seed

        if isinstance(activation, type):
            act_kwargs = {"negative_slope": 0.2}
            self.activation = activation(**act_kwargs)  # instantiate
        else:
            self.activation = activation  # already a module/callable

        self._compute_base_sizes()
        self._build_modules()

    @property
    def shape(self):
        return (1, int(self.env.args['nely']), int(self.env.args['nelx']))

    def _compute_base_sizes(self):
        total_up = 1
        for f in self.conv_upsample:
            total_up *= int(f)

        H, W = self.env.args['nely'], self.env.args['nelx']

        if H % total_up != 0 or W % total_up != 0:
            raise ValueError(f"Grid {H}x{W} too small,not divisible by Π(conv_upsample)={total_up}")

        self.h_base = H // total_up
        self.w_base = W // total_up
        self._target_hw = (H, W)

        self.conv_stage_sizes = []
        cur_h, cur_w = self.h_base, self.w_base
        for f in self.conv_upsample:
            cur_h *= int(f); cur_w *= int(f)
            self.conv_stage_sizes.append((cur_h, cur_w))

    def _build_modules(self):   # <- consistent name
        # latent + dense
        self.latent = nn.Parameter(torch.randn(1, self.latent_size))
        self.dense  = nn.Linear(self.latent_size, self.h_base * self.w_base * self.init_channels)
        _init_dense_orthogonal(self.dense, self.latent_size, self.h_base, self.w_base, self.init_channels)        

        # convs + offsets
        conv_blocks, offsets = [], []
        in_ch = self.init_channels
        for out_ch in self.conv_filters:
            conv_blocks.append(ConvBlock(in_ch, int(out_ch), k=self.kernel_size))
            offsets.append(AddOffset(channels=int(out_ch), scale=self.offset_scale))
            in_ch = int(out_ch)
        self.conv_blocks  = nn.ModuleList(conv_blocks)
        self.offsets = nn.ModuleList(offsets)

        self.final_1x1 = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

        # init: Kaiming for convs except final_1x1
        neg = 0.2  # Leaky slope
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.final_1x1:
                nn.init.kaiming_normal_(m.weight, a=neg, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.final_1x1.weight.zero_() 
            self.final_1x1.bias.fill_(logit(self.problem_params.density))  # logit target density

    def _downsample_for_distill(self, logits: torch.Tensor) -> torch.Tensor:
        # logits_f32: (1,H,W) or (H,W)
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)  # (1,H,W)
        x = logits.unsqueeze(0)               # (1,1,H,W)
        s = int(getattr(self, "_distill_scale", 1))
        if s > 1:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=s, stride=s)
        return x[0,0]

    def _set_warmstart_strength(self, tau: float):
        self._warm_tau = float(min(1.0, max(0.0, tau)))

    @torch.no_grad()
    def upsample(self, scale=2, distill_weight=0.1,freeze_transferred=True):

        if not isinstance(scale, int) or scale < 2:
            raise ValueError("scale must be an integer >= 2")
        
        # set up teacher and run once
        with torch.no_grad():
            teacher_target = self.forward()[0].to(torch.float32).contiguous()
            rho_lo = torch.sigmoid(teacher_target)  # (H_old, W_old)
            rho_hi = F.interpolate(
                rho_lo[None, None],  # (1,1,H_old,W_old)
                scale_factor=scale,
                mode='bilinear', align_corners=False
            )[0, 0]  # (H_new, W_new)
            rho_hi = rho_hi.clamp_(0.01, 0.99)
            rho_hi_logit = torch.log(rho_hi) - torch.log1p(-rho_hi)
            self.rho_hi_logit = rho_hi_logit

        _, H, W = self.shape
        new_H, new_W = H * scale, W * scale
        self._update_problem_params({'height': new_H, 'width': new_W})

        ups = list(map(int, self.conv_upsample))
        ups[-1] *= int(scale)
        self.conv_upsample = tuple(ups)
        
        self._compute_base_sizes()

        # register buffers/knobs
        self.register_buffer("_rho_hi_logit", rho_hi_logit, persistent=False)
        self.register_buffer("_distill_target", teacher_target, persistent=False)
        self._distill_scale = int(scale)
        self._distill_weight = float(distill_weight)

        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self):
        # [1, C, h_base, w_base] in fp32
        z = self.latent.float()
        x = self.dense(z).view(1, self.init_channels, self.h_base, self.w_base)

        # upsample conv stages
        for (h_target, w_target), conv_block, offset in zip(self.conv_stage_sizes, self.conv_blocks, self.offsets):
            x = self.activation(x)
            x = _upsample_to(x, (h_target, w_target), mode="nearest")
            x = conv_block(x)
            x = offset(x)

        x = self.final_1x1(x)

        tau = float(getattr(self, "_warm_tau", 0.0))
        if tau > 0.0 and hasattr(self, "_rho_hi_logit"):
            if self._rho_hi_logit.shape[-2:] == x.shape[-2:]:
                # x ← (1-τ)x + τ x*
                x = (1.0 - tau) * x + tau * self._rho_hi_logit[None, None]

        
        if self.final_activation is not None:
            x = self.final_activation(x)

        # (1,1,H,W) -> (1,H,W) as float64 for physics
        x = x[:, 0, :, :].to(torch.float64)

        return x

    def loss(self, logits=None):
        if logits is None:
            logits = self.forward()
        base_loss = self.get_total_loss(logits)

        distill_weight = float(getattr(self, "_distill_weight", 0.0))
        distill_target = getattr(self, "_distill_target", None)
        distill_scale= int(getattr(self, "_distill_scale", 1))

        if distill_weight > 0.0 and distill_target is not None and distill_scale >= 1:
            student_downsample = self._downsample_for_distill(logits.to(torch.float32))  # (H_old, W_old)
            distill = torch.nn.functional.mse_loss(student_downsample, distill_target)
            base_loss = base_loss + distill_weight * distill

        return base_loss
