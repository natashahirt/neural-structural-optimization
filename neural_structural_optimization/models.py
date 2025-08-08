import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from neural_structural_optimization.models_utils import (
    set_random_seed, batched_topo_loss
)
from neural_structural_optimization.problems_utils import ProblemParams
from neural_structural_optimization import topo_api

# Optional: wire in your CLIP when ready
try:
    from neural_structural_optimization.clip_loss import CLIPLoss
except Exception:
    CLIPLoss = None


# ---------- NumPy/Autograd bridge for physics ----------

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
    def __init__(self, problem_params=None, clip_config=None, seed=None, args=None):
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

        self.clip_loss = None
        self.clip_weight = 0.0
        if clip_config is not None and getattr(clip_config, "weight", 0) > 0 and CLIPLoss is not None:
            self.clip_loss = CLIPLoss(config=clip_config)
            self.clip_weight = float(clip_config.weight)

    def forward(self):
        # subclasses define self.z or a generator
        raise NotImplementedError

    def physics_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # Returns mean over batch of structural losses
        per_example = StructuralLoss.apply(logits, self.env)  # (batch,)
        return per_example.mean()

    def total_loss(self, logits: torch.Tensor) -> torch.Tensor:
        loss = self.physics_loss(logits)
        if self.clip_loss is not None:
            # NOTE: keep CLIP on float32 normally, then cast
            if self.clip_loss.target_text_prompt is not None:
                clip = self.clip_loss.get_text_loss(logits, self.clip_loss.target_text_prompt)
                clip = clip.mean().to(loss.dtype)
                loss = loss + self.clip_weight * clip
        return loss

    def update_problem_params(self, new_params):
        if isinstance(new_params, dict):
            new_params = ProblemParams(**new_params)
        self.problem_params = new_params
        problem = new_params.get_problem()
        new_args = topo_api.specified_task(problem)
        self.env = topo_api.Environment(new_args)
        self.args = new_args
        return new_args


# ========================================
# Pixel model
# ========================================

class PixelModel(Model):
    def __init__(self, problem_params=None, clip_config=None, seed=None):
        super().__init__(problem_params=problem_params, clip_config=clip_config, seed=seed)
        # shape: (batch=1, H, W)
        H, W = self.env.args['nely'], self.env.args['nelx']
        self.shape = (1, H, W)
        z_init = np.broadcast_to(self.env.args['volfrac'] * self.env.args['mask'], self.shape)
        self.z = nn.Parameter(torch.tensor(z_init, dtype=torch.float64), requires_grad=True)

    def forward(self):
        return self.z

    def loss(self):
        logits = self.forward()
        return self.total_loss(logits)


# ========================================
# CNN
# ========================================

def _init_kaiming(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, act=nn.LeakyReLU, add_offset_scale=0.0, norm=None):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad)
        self.norm = norm(out_ch) if norm is not None else None
        self.act = act()
        self.offset = nn.Parameter(torch.zeros(1, out_ch, 1, 1)) if add_offset_scale != 0 else None
        self.scale = add_offset_scale
        self.apply(_init_kaiming)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        if self.offset is not None:
            x = x + self.scale * self.offset
        return x

class CNNModel(Model):
    def __init__(
        self,
        problem_params=None,
        clip_config=None,
        latent_size=128,
        dense_channels=32,
        upsample_factors=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16, 1),
        offset_scale=10.0,
        kernel_size=5,
        act=nn.LeakyReLU,    # pass a class
        norm=None,           # e.g., nn.BatchNorm2d or lambda C: nn.GroupNorm(8, C)
        final_activation=None,
        upsample_mode="nearest", 
    ):
        super().__init__(problem_params=problem_params, clip_config=clip_config)

        if len(upsample_factors) != len(conv_filters):
            raise ValueError("upsample_factors and conv_filters must have same length")

        H, W = self.env.args['nely'], self.env.args['nelx']
        total_up = 1
        for f in upsample_factors:
            total_up *= int(f)

        if H % total_up != 0 or W % total_up != 0:
            raise ValueError(f"Grid {H}x{W} not divisible by Π(upsample_factors)={total_up}")

        # Store coarse grid dims and channels for reshape
        self.h0 = H // total_up
        self.w0 = W // total_up
        self.init_channels = int(dense_channels)
        self._target_hw = (H, W)
        self.upsample_mode = upsample_mode

        # latent and dense
        self.latent = nn.Parameter(torch.randn(1, latent_size))
        self.dense = nn.Linear(latent_size, self.h0 * self.w0 * self.init_channels)
        _init_kaiming(self.dense)

        # Precompute exact per-stage target sizes
        self.stage_sizes = []
        cur_h, cur_w = self.h0, self.w0
        for f in upsample_factors:
            f = int(f)
            cur_h *= f
            cur_w *= f
            self.stage_sizes.append((cur_h, cur_w))

        # Conv tower (no layers created in forward)
        blocks = []
        in_ch = self.init_channels
        for out_ch in conv_filters:
            blocks.append(ConvBlock(in_ch, int(out_ch), k=kernel_size, act=act, add_offset_scale=offset_scale, norm=norm))
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
        x = self.dense(self.latent.float()).view(1, self.init_channels, self.h0, self.w0)

        # nearest-neighbor upsample → conv block
        for (Ht, Wt), block in zip(self.stage_sizes, self.blocks):
            if (x.shape[-2], x.shape[-1]) != (Ht, Wt):
                x = F.interpolate(x, size=(Ht, Wt), mode=self.upsample_mode)  # << nearest
            x = block(x)

        x = self.final_1x1(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        # (1,1,H,W) → (1,H,W) fp64 for physics
        return x[:, 0, :, :].to(torch.float64)

    def loss(self):
        return self.total_loss(self.forward())
