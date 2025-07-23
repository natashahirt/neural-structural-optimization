# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=missing-docstring
# pylint: disable=superfluous-parens
import functools

from absl import logging
import autograd
import autograd.numpy as np
from neural_structural_optimization import models
from neural_structural_optimization import topo_physics
import scipy.optimize
import tensorflow as tf
import xarray


def optimizer_result_dataset(losses, frames, save_intermediate_designs=False):
  """Create an xarray dataset from optimization results.
  
  Args:
    losses: Array of loss values from optimization steps
    frames: Array of design frames from optimization steps
    save_intermediate_designs: Whether to save all intermediate designs
    
  Returns:
    xarray.Dataset containing the optimization results
  """
  # The best design will often but not always be the final one.
  best_design = np.nanargmin(losses)
  logging.info(f'Final loss: {losses[best_design]}')
  if save_intermediate_designs:
    ds = xarray.Dataset({
        'loss': (('step',), losses),
        'design': (('step', 'y', 'x'), frames),
    }, coords={'step': np.arange(len(losses))})
  else:
    ds = xarray.Dataset({
        'loss': (('step',), losses),
        'design': (('y', 'x'), frames[best_design]),
    }, coords={'step': np.arange(len(losses))})
  return ds


def train_tf_optimizer(
    model, max_iterations, optimizer, save_intermediate_designs=True,
):
  """Train a model using TensorFlow optimizers.
  
  Args:
    model: The model to train
    max_iterations: Maximum number of optimization iterations
    optimizer: TensorFlow optimizer to use
    save_intermediate_designs: Whether to save intermediate designs
    
  Returns:
    xarray.Dataset containing optimization results
  """
  loss = 0
  model(None)  # build model, if not built
  tvars = model.trainable_variables

  losses = []
  frames = []
  for i in range(max_iterations + 1):
    with tf.GradientTape() as t:
      t.watch(tvars)
      logits = model(None)
      loss = model.loss(logits)

    losses.append(loss.numpy().item())
    frames.append(logits.numpy())

    if i % (max_iterations // 10) == 0:
      logging.info(f'step {i}, loss {losses[-1]:.2f}')

    if i < max_iterations:
      grads = t.gradient(loss, tvars)
      optimizer.apply_gradients(zip(grads, tvars))

  designs = [model.env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(np.array(losses), np.array(designs),
                                  save_intermediate_designs)


train_adam = functools.partial(
    train_tf_optimizer, optimizer=tf.keras.optimizers.Adam(1e-2))


def _set_variables(variables, x):
  """Set TensorFlow variables from a flattened array.
  
  Args:
    variables: List of TensorFlow variables
    x: Flattened array of values to assign
  """
  # Use shape.as_list() for compatibility with modern TensorFlow
  shapes = [list(v.shape) for v in variables]
  values = tf.split(x, [np.prod(s) for s in shapes])
  for var, value in zip(variables, values):
    var.assign(tf.reshape(tf.cast(value, var.dtype), var.shape))


def _get_variables(variables):
  """Get flattened array from TensorFlow variables.
  
  Args:
    variables: List of TensorFlow variables
    
  Returns:
    Flattened numpy array of variable values
  """
  return np.concatenate([
      v.numpy().ravel() if not isinstance(v, np.ndarray) else v.ravel()
      for v in variables])


def train_lbfgs(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
    **kwargs
):
  """Train a model using L-BFGS optimization.
  
  Args:
    model: The model to train
    max_iterations: Maximum number of optimization iterations
    save_intermediate_designs: Whether to save intermediate designs
    init_model: Optional model to initialize from
    **kwargs: Additional arguments for scipy.optimize.fmin_l_bfgs_b
    
  Returns:
    xarray.Dataset containing optimization results
  """
  model(None)  # build model, if not built

  losses = []
  frames = []

  if init_model is not None:
    if not isinstance(model, models.PixelModel):
      raise TypeError('can only use init_model for initializing a PixelModel')
    model.z.assign(tf.cast(init_model(None), model.z.dtype))

  tvars = model.trainable_variables

  def value_and_grad(x):
    _set_variables(tvars, x)
    with tf.GradientTape() as t:
      t.watch(tvars)
      logits = model(None)
      loss = model.loss(logits)
    grads = t.gradient(loss, tvars)
    frames.append(logits.numpy().copy())
    losses.append(loss.numpy().copy())
    return float(loss.numpy()), _get_variables(grads).astype(np.float64)

  x0 = _get_variables(tvars).astype(np.float64)
  # rely upon the step limit instead of error tolerance for finishing.
  _, _, info = scipy.optimize.fmin_l_bfgs_b(
      value_and_grad, x0, maxfun=max_iterations, factr=1, pgtol=1e-14, **kwargs
  )
  logging.info(info)

  designs = [model.env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(
      np.array(losses), np.array(designs), save_intermediate_designs)


def constrained_logits(init_model):
  """Produce matching initial conditions with volume constraints applied."""
  logits = init_model(None).numpy().astype(np.float64).squeeze(axis=0)
  return topo_physics.physical_density(
      logits, init_model.env.args, volume_contraint=True, cone_filter=False)


def method_of_moving_asymptotes(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
):
  """Train a model using Method of Moving Asymptotes (MMA) optimization.
  
  Args:
    model: The model to train (must be PixelModel)
    max_iterations: Maximum number of optimization iterations
    save_intermediate_designs: Whether to save intermediate designs
    init_model: Optional model to initialize from
    
  Returns:
    xarray.Dataset containing optimization results
  """
  import nlopt  # pylint: disable=g-import-not-at-top

  if not isinstance(model, models.PixelModel):
    raise ValueError('MMA only defined for pixel models')

  env = model.env
  if init_model is None:
    x0 = _get_variables(model.trainable_variables).astype(np.float64)
  else:
    x0 = constrained_logits(init_model).ravel()

  def objective(x):
    return env.objective(x, volume_contraint=False)

  def constraint(x):
    return env.constraint(x)

  def wrap_autograd_func(func, losses=None, frames=None):
    def wrapper(x, grad):
      if grad.size > 0:
        value, grad[:] = autograd.value_and_grad(func)(x)
      else:
        value = func(x)
      if losses is not None:
        losses.append(value)
      if frames is not None:
        frames.append(env.reshape(x).copy())
      return value
    return wrapper

  losses = []
  frames = []

  opt = nlopt.opt(nlopt.LD_MMA, x0.size)
  opt.set_min_objective(wrap_autograd_func(objective, losses, frames))
  opt.add_inequality_constraint(wrap_autograd_func(constraint))
  opt.set_lower_bounds(0.001)
  opt.set_upper_bounds(1.0)

  try:
    x = opt.optimize(x0)
    logging.info(f'MMA optimization completed successfully')
  except nlopt.RoundoffLimited:
    logging.info('MMA optimization stopped due to roundoff errors')
  except nlopt.MaxEvalReached:
    logging.info('MMA optimization stopped due to maximum evaluations')
  except nlopt.MaxTimeReached:
    logging.info('MMA optimization stopped due to time limit')

  designs = [env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(
      np.array(losses), np.array(designs), save_intermediate_designs)


def optimality_criteria(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
):
  """Train a model using Optimality Criteria optimization.
  
  Args:
    model: The model to train (must be PixelModel)
    max_iterations: Maximum number of optimization iterations
    save_intermediate_designs: Whether to save intermediate designs
    init_model: Optional model to initialize from
    
  Returns:
    xarray.Dataset containing optimization results
  """
  if not isinstance(model, models.PixelModel):
    raise ValueError('Optimality criteria only defined for pixel models')

  env = model.env
  if init_model is None:
    x = _get_variables(model.trainable_variables).astype(np.float64)
  else:
    x = constrained_logits(init_model).ravel()

  losses = []
  frames = []

  for i in range(max_iterations):
    x = topo_physics.optimality_criteria_step(x, env.ke, env.args)
    loss = env.objective(x, volume_contraint=False)
    losses.append(loss)
    frames.append(env.reshape(x).copy())

    if i % (max_iterations // 10) == 0:
      logging.info(f'step {i}, loss {loss:.2f}')

  designs = [env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(
      np.array(losses), np.array(designs), save_intermediate_designs)


def train_batch(model_list, flag_values, train_func=train_adam):
  """Train multiple models in batch.
  
  Args:
    model_list: List of models to train
    flag_values: Configuration flags
    train_func: Training function to use
    
  Returns:
    List of optimization results
  """
  results = []
  for model in model_list:
    result = train_func(model, flag_values.optimization_steps)
    results.append(result)
  return results
