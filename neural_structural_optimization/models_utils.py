import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topo_api, pipeline_utils, models_utils, problems_utils
import tensorflow as tf
import torch

layers = tf.keras.layers

# =============================================================================
# Loss functions
# =============================================================================

def batched_topo_loss(params, envs):
  losses = [env.objective(params[i], volume_constraint=True)
            for i, env in enumerate(envs)]
  return np.stack(losses)

# =============================================================================
# Autodiff integration utilities
# =============================================================================

def convert_autograd_to_tensorflow(func):
  @tf.custom_gradient
  def wrapper(x):
    vjp, ans = autograd.core.make_vjp(func, x.numpy())
    return ans, vjp
  return wrapper

# =============================================================================
# Utility functions
# =============================================================================

def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
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
# Layer factory functions
# =============================================================================

def UpSampling2D(factor):
  return layers.UpSampling2D((factor, factor), interpolation='bilinear')

def Conv2D(filters, kernel_size, **kwargs):
  return layers.Conv2D(filters, kernel_size, padding='same', **kwargs)

# =============================================================================
# Custom layers
# =============================================================================

class AddOffset(layers.Layer):

  def __init__(self, scale=1):
    super().__init__()
    self.scale = scale

  def build(self, input_shape):
    self.bias = self.add_weight(
        shape=input_shape, initializer='zeros', trainable=True, name='bias')

  def call(self, inputs):
    return inputs + self.scale * self.bias
