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
import matplotlib.pyplot as plt
from neural_structural_optimization import models  # Add this import
from neural_structural_optimization import topo_physics, topo_api
import scipy.optimize
import tensorflow as tf
import xarray
from tqdm import tqdm


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
    train_tf_optimizer, optimizer=tf.keras.optimizers.legacy.Adam(1e-2))


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
    # if not isinstance(model, models.PixelModel):
      # raise TypeError('can only use init_model for initializing a PixelModel')
    if not hasattr(model, 'z'):
      raise TypeError('init_model can only be used for models with latent variable `z`')
    model.z.assign(tf.cast(init_model(None), model.z.dtype))

  tvars = model.trainable_variables
  pbar = tqdm(total=max_iterations)

  def value_and_grad(x):
    pbar.update(1)
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

def adaptive_train_lbfgs(
  problem,
  resolutions,
  max_iter_per_stage,
  model_class=models.CNNModelDynamic,
  **kwargs
):
  z_init = None # initial latent vector
  old_shape = None
  old_dense = None
  ds_history = []
  
  for i, (nelx, nely) in enumerate(resolutions):
    print(f"\n=== Stage {i+1}/{len(resolutions)}: {nely} x {nelx} ===")
    # Create a fresh problem object for each stage instead of modifying the original
    clean_problem = problem.copy(width=nelx, height=nely)
    args = topo_api.specified_task(clean_problem)

    model = model_class(args=args)

    if z_init is not None: # project the existing latent vector
      # old_x, old_y = resolutions[i-1]
      old_x, old_y = old_shape
      new_x, new_y = model.base_shape

      print("z init shape", z_init.shape)

      # old_c = z_init.shape[1] // (old_x * old_y)
      # new_c = model.z.shape[1] // (new_x * new_y)

      old_c = old_dense
      new_c = model.dense_channels

      print("old base shape: ", old_shape)
      print("old dense channels: ", old_dense)
      print("old x:", old_x, " old y:", old_y, " old:", old_c)
      print("new x:", resolutions[i][0], " new y:", resolutions[i][1], " new_c:", new_c)
      print("new base shape: ", model.base_shape)
      print("new dense channels:", model.dense_channels)

      z_spatial = tf.reshape(z_init, [1, old_y, old_x, old_c])
      z_resized = tf.image.resize(z_spatial, size=(new_y, new_x), method='bilinear')

      if old_c != new_c:
        # Use a 1x1 conv to project channels if needed
        conv_projection_layer = tf.keras.layers.Conv2D(new_c, kernel_size=1, use_bias=False)
        conv_projection_layer.build(z_resized.shape)
        W = tf.eye(new_c, num_columns=old_c)  # shape (new_c, old_c)
        W = tf.reshape(W, [1, 1, old_c, new_c])  # for Conv2D
        conv_projection_layer.set_weights([W.numpy()])
        z_resized = conv_projection_layer(z_resized)
      
      z_final = tf.reshape(z_resized, [1, new_y * new_x * new_c])
      model.z.assign(z_final)
    
    ds = train_lbfgs(model, max_iter_per_stage[i])
    ds_history.append(ds)
    
    z_init = model.z.numpy()
    old_shape = model.base_shape
    old_dense = model.dense_channels
  
  return ds_history

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
    """Train a model using Optimality Criteria optimization."""
    from neural_structural_optimization import models
    
    if not isinstance(model, models.PixelModel):
        raise ValueError('Optimality criteria only defined for pixel models')

    env = model.env
    nely, nelx = env.args['nely'], env.args['nelx']
    expected_size = nely * nelx

    # Initialize design
    if init_model is None:
        x = _get_variables(model.trainable_variables).astype(np.float64)
    else:
        x = constrained_logits(init_model).ravel()

    # Ensure x is 1D array with correct size
    x = np.asarray(x).ravel()
    if x.size != expected_size:
        logging.warning(f'Reshaping x from {x.size} to {expected_size}')
        if x.size == 1:
            x = np.full(expected_size, float(x[0]))
        else:
            x = x[:expected_size]
            if x.size < expected_size:
                x = np.pad(x, (0, expected_size - x.size), mode='edge')

    losses = []
    frames = []

    for i in range(max_iterations):
        try:
            # Apply optimality criteria step
            step_result = topo_physics.optimality_criteria_step(x, env.ke, env.args)
            
            # Handle return value - ensure it's a 1D array
            if isinstance(step_result, tuple):
                x_new = step_result[0]
            else:
                x_new = step_result
            
            # Convert to numpy array and ensure it's 1D
            x_new = np.asarray(x_new)
            if x_new.ndim == 0:  # Scalar
                x_new = np.full(expected_size, float(x_new))
            else:
                x_new = x_new.ravel()
            
            # Ensure correct size
            if x_new.size != expected_size:
                logging.warning(f'Step {i}: result size {x_new.size} != expected {expected_size}')
                if x_new.size == 1:
                    x_new = np.full(expected_size, float(x_new[0]))
                else:
                    x_new = x_new[:expected_size]
                    if x_new.size < expected_size:
                        x_new = np.pad(x_new, (0, expected_size - x_new.size), mode='edge')
            
            x = x_new
            
            # Calculate loss
            loss = env.objective(x, volume_contraint=False)
            losses.append(loss)
            
            # Create frame
            frame = x.reshape(nely, nelx)
            frames.append(frame.copy())

            if i % max(1, max_iterations // 10) == 0:
                logging.info(f'step {i}, loss {loss:.6f}')
                
        except Exception as e:
            logging.warning(f'Step {i} failed: {e}')
            break

    # Ensure we have results
    if not losses:
        losses = [0.0]
        frames = [np.zeros((nely, nelx))]

    # Create designs
    designs = []
    for frame in frames:
        try:
            design = env.render(frame, volume_contraint=True)
        except:
            design = frame
        designs.append(design)

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
