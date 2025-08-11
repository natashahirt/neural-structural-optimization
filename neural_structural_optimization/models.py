import copy

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

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


# ---------- NumPy/Autograd bridge for physics ----------

def _gauss33(device, dtype):
    k = torch.tensor([[1,2,1],
                      [2,4,2],
                      [1,2,1]], dtype=dtype, device=device)
    k = (k / k.sum()).view(1,1,3,3)        # [C_out=1,C_in=1,H,W]
    return k

def _logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

def _tv_loss(x):
    # x in [0,1], shape [B,1,H,W] or [B,3,H,W]
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

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
        
        self.clip_loss = clip_loss
        if clip_loss is not None and getattr(clip_loss, "weight", 0) > 0:
            self.clip_weight = float(clip_loss.weight)

    def forward(self):
        # subclasses define self.z or a generator
        raise NotImplementedError

    def get_physics_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # Returns mean over batch of structural losses
        per_example = StructuralLoss.apply(logits, self.env)  # (batch,)
        return per_example.mean()
        
    def get_total_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """if clip_weight is None:
            clip_weight = getattr(self.clip_loss, "weight", 0.0)
        if tv_weight is None:
            tv_weight = 1e-3
        total_loss = physics_loss

        if self.clip_loss is not None:
            clip_loss = self.clip_loss.get_text_loss(logits).mean().to(physics_loss.dtype)
            total_loss = clip_weight * clip_loss"""
        
        return self.get_physics_loss(logits)

    def update_problem_params(self, new_params):
        self.problem_params = self.problem_params.copy(**new_params)
        problem = self.problem_params.get_problem()
        new_args = topo_api.specified_task(problem)
        self.env = topo_api.Environment(new_args)
        self.args = new_args
        return new_args


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
        H, W = self.env.args['nely'], self.env.args['nelx'] # shape: (batch=1, H, W)
        # init as densities
        z_init = np.broadcast_to(self.env.args['volfrac'] * self.env.args['mask'], self.shape)
        self.z = nn.Parameter(torch.tensor(z_init, dtype=torch.float64), requires_grad=True)

    @property
    def shape(self):
        return (1, self.env.args['nely'], self.env.args['nelx'])

    def forward(self):
        return self.z

    def loss(self, logits=None, clip_weight=None, tv_weight=None):
        logits = self.forward()
        return self.get_total_loss(logits, clip_weight, tv_weight)

    def upsample(self, scale=2, soften=True, preserve_mean=True):
        current_z = self.z.detach()

        # update problem
        self.problem_params.width *= scale
        self.problem_params.height *= scale
        
        problem = self.problem_params.get_problem()
        new_args = topo_api.specified_task(problem)
        self.env = topo_api.Environment(new_args)
        self.args = new_args
        
        # update model
        new_height = int(current_z.shape[-2] * scale)
        new_width = int(current_z.shape[-1] * scale)

        resized_z = F.interpolate(
            current_z.unsqueeze(0) if current_z.dim() == 3 else current_z,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )

        if current_z.dim() == 3:
            resized_z = resized_z.squeeze(0)

        # replace parameter
        self.z = torch.nn.Parameter(resized_z, requires_grad=True)

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
        conv_filters=(128, 64, 32, 16, 1),
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
    
    @property
    def shape(self):
        return (1, self.env.args['nely'], self.env.args['nelx'])

    @torch.no_grad()
    def upsample(self, scale=2, dynamic_depth_fn=None, freeze_transferred=True):
        old_state = copy.deepcopy(self.state_dict())

        new_W = int(self.problem_params.width  * scale)
        new_H = int(self.problem_params.height * scale)

        self.update_problem_params({"width": new_W, "height": new_H})

        if dynamic_depth_fn is not None:
            # {'conv_upsample': tuple, 'conv_filters': tuple, 'init_channels': int}
            depth = dynamic_depth_fn(self.problem_params)
            if 'conv_upsample' in depth: self.conv_upsample = tuple(int(x) for x in depth['conv_upsample'])
            if 'conv_filters'  in depth: self.conv_filters  = tuple(int(x) for x in depth['conv_filters'])
            if 'init_channels' in depth: self.init_channels = int(depth['init_channels'])

        self._compute_base_sizes()
        self._build_modules()

        # 5) transfer-by-shape
        new_state = self.state_dict()
        transferred = []
        for k, v in new_state.items():
            if k in old_state and old_state[k].shape == v.shape:
                v.copy_(old_state[k])
                transferred.append(k)
        self.load_state_dict(new_state, strict=False)

        if freeze_transferred:
            frozen = set(transferred)
            for name, p in self.named_parameters():
                if name in frozen:
                    p.requires_grad_(False)

        # 7) return info (for logging)
        return transferred

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
        
        if self.final_activation is not None:
            x = self.final_activation(x)

        # (1,1,H,W) -> (1,H,W) as float64 for physics
        return x[:, 0, :, :].to(torch.float64)

    def loss(self, logits=None):
        if logits is None:
            logits = self.forward()
        return self.get_total_loss(logits.to(torch.float64))
