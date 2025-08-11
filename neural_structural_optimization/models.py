import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from neural_structural_optimization.models_utils import (
    set_random_seed, batched_topo_loss, to_nchw, normalize_mask
)
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
        if isinstance(new_params, dict):
            new_params = ProblemParams(**new_params)
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


def _init_kaiming(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, activation=nn.LeakyReLU, add_offset_scale=0.0, norm=None):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad)
        self.norm = norm(out_ch) if norm is not None else None
        self.activation = activation()
        self.offset = nn.Parameter(torch.zeros(1, out_ch, 1, 1)) if add_offset_scale != 0 else None
        self.scale = add_offset_scale
        self.apply(_init_kaiming)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        if self.offset is not None:
            x = x + self.scale * self.offset
        return x

class CNNModel(Model):
    def __init__(
        self,
        problem_params=None,
        latent_size=128,
        init_channels=32,
        conv_upsample=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16, 1),
        offset_scale=10.0,
        kernel_size=5,
        activation=nn.LeakyReLU,    # pass a class
        norm=None,           # e.g., nn.BatchNorm2d or lambda C: nn.GroupNorm(8, C)
        final_activation=None,
    ):
        super().__init__(problem_params=problem_params)

        if len(conv_upsample) != len(conv_filters):
            raise ValueError("conv_upsample and conv_filters must have same length")

        H, W = self.env.args['nely'], self.env.args['nelx']
        total_up = 1
        for f in conv_upsample:
            total_up *= int(f)

        if H % total_up != 0 or W % total_up != 0:
            raise ValueError(f"Grid {H}x{W} not divisible by Π(conv_upsample)={total_up}")

        # Store coarse grid dims and channels for reshape
        self.h0 = H // total_up
        self.w0 = W // total_up
        self.init_channels = int(init_channels)
        self._target_hw = (H, W)

        # latent and dense
        self.latent = nn.Parameter(torch.randn(1, latent_size))
        self.dense = nn.Linear(latent_size, self.h0 * self.w0 * self.init_channels)
        _init_kaiming(self.dense)

        # Precompute exact per-stage target sizes
        self.stage_sizes = []
        cur_h, cur_w = self.h0, self.w0
        for f in conv_upsample:
            f = int(f)
            cur_h *= f
            cur_w *= f
            self.stage_sizes.append((cur_h, cur_w))

        # Conv tower (no layers created in forward)
        blocks = []
        in_ch = self.init_channels
        for out_ch in conv_filters:
            blocks.append(ConvBlock(in_ch, int(out_ch), k=kernel_size, activation=activation, add_offset_scale=offset_scale, norm=norm))
            in_ch = int(out_ch)
        self.blocks = nn.ModuleList(blocks)

        self.final_1x1 = nn.Conv2d(in_ch, 1, kernel_size=1)
        _init_kaiming(self.final_1x1)

        self.final_activation = final_activation() if final_activation is not None else None

    @property
    def shape(self):
        return (1, self.env.args['nely'], self.env.args['nelx'])

    def forward(self):
        # latent -> dense -> [1, C, h0, w0] in fp32
        z = self.latent.float()
        x = self.dense(z).view(1, self.init_channels, self.h0, self.w0)

        # Upsample to each stage size and apply the corresponding block
        for (Ht, Wt), block in zip(self.stage_sizes, self.blocks):
            if (x.shape[-2], x.shape[-1]) != (Ht, Wt):
                x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
            x = block(x)

        x = self.final_1x1(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        # (1,1,H,W) -> (1,H,W) as float64 for physics
        return x[:, 0, :, :].to(torch.float64)

    def loss(self):
        return self.get_total_loss(self.forward())
