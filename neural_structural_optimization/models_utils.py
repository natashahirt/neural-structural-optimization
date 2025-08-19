import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topo_api, pipeline_utils, models_utils, problems_utils
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Loss functions
# =============================================================================

def batched_topo_loss(params, envs):
  losses = [env.objective(params[i], volume_constraint=True)
            for i, env in enumerate(envs)]
  return np.stack(losses)

# =============================================================================
# Utility functions
# =============================================================================

def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

def global_normalization(inputs, epsilon=1e-6):
  """PyTorch equivalent of TensorFlow's global normalization."""
  if isinstance(inputs, torch.Tensor):
    mean = inputs.mean()
    variance = inputs.var()
    net = inputs - mean
    net = net * torch.rsqrt(variance + epsilon)
    return net
  else:
    # Fallback for numpy arrays
    mean = np.mean(inputs)
    variance = np.var(inputs)
    net = inputs - mean
    net = net * np.sqrt(1.0 / (variance + epsilon))
    return net

def to_nchw(z):
    # expects (1,H,W) or (H,W)
    if z.ndim == 2:         # (H,W)
        z = z.unsqueeze(0)  # (1,H,W)
    if z.ndim == 3:         # (1,H,W)
        z = z.unsqueeze(1)  # (1,1,H,W)
    assert z.ndim == 4, f"bad shape: {z.shape}"
    return z

def normalize_mask(m, H, W, device, dtype):
    """Return a [H,W] mask tensor (float {0,1})."""
    if m is None:
        return torch.ones((H, W), device=device, dtype=dtype)

    m = torch.as_tensor(m, device=device, dtype=dtype)

    # Scalar -> broadcast
    if m.numel() == 1:
        val = float(m.item())
        return torch.full((H, W), val, device=device, dtype=dtype)

    # 1D flat -> reshape if size matches
    if m.dim() == 1 and m.numel() == H * W:
        return m.view(H, W)

    # Already [H,W]
    if m.dim() == 2 and m.shape == (H, W):
        return m

    # Anything else -> resize with NEAREST
    if m.dim() == 2:  # different size
        m4 = m.unsqueeze(0).unsqueeze(0)  # [1,1,h0,w0]
        return F.interpolate(m4, size=(H, W), mode="nearest")[0, 0]

    # Last resort (unexpected shape)
    m4 = m.view(1, 1, *([1] * max(0, 2 - m.dim())))  # coerce to 4D-ish
    return F.interpolate(m4, size=(H, W), mode="nearest")[0, 0]

# =============================================================================
# Layer factory functions (PyTorch equivalents)
# =============================================================================

def UpSampling2D(factor):
  """PyTorch equivalent of TensorFlow's UpSampling2D."""
  return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

def Conv2D(filters, kernel_size, **kwargs):
  """PyTorch equivalent of TensorFlow's Conv2D."""
  padding = kwargs.get('padding', 'same')
  if padding == 'same':
    padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
  
  return nn.Conv2d(
    in_channels=kwargs.get('input_shape', [1])[-1] if 'input_shape' in kwargs else 1,
    out_channels=filters,
    kernel_size=kernel_size,
    padding=padding,
    **{k: v for k, v in kwargs.items() if k not in ['padding', 'input_shape']}
  )

# =============================================================================
# Custom layers
# =============================================================================

class AddOffset(nn.Module):
  def __init__(self, channels, scale=10.0):
      super().__init__()
      self.scale = float(scale)
      # fixed shape: (1, C, 1, 1) — transfers across H×W changes
      self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
      
  def forward(self, x):
      return x + self.scale * self.bias