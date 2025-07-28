import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topo_api, pipeline_utils, models_utils, problems_utils
import tensorflow as tf

layers = tf.keras.layers

# =============================================================================
# Loss functions
# =============================================================================

def batched_topo_loss(params, envs):
  losses = [env.objective(params[i], volume_contraint=True)
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