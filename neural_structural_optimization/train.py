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

import torch
import numpy as np

# utilities

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

def constrained_logits(init_model):
  """Produce matching initial conditions with volume constraints applied."""
  logits = init_model(None).numpy().astype(np.float64).squeeze(axis=0)
  return topo_physics.physical_density(
      logits, init_model.env.args, volume_constraint=True, cone_filter=False)


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

# training

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

  designs = [model.env.render(x, volume_constraint=True) for x in frames]
  return optimizer_result_dataset(np.array(losses), np.array(designs),
                                  save_intermediate_designs)

def _cosine_warmup(t, T, warmup=0.1, start=1.0, end=0.0):
    """Cosine from `start`→`end` after a linear warmup portion."""
    Tw = max(int(T * warmup), 1)
    if t < Tw:
        return start * (t + 1) / Tw
    tt = (t - Tw) / max(T - Tw, 1)
    return end + 0.5 * (start - end) * (1 + np.cos(np.pi * tt))

def train_adam(model,
    max_iterations,
    lr_init=1e-2,
    lr_final=3e-3,
    clip_w_init=0.5,         # strong early CLIP to form silhouette
    clip_w_final=0.08,       # decay to weaker guidance
    tv_w_init=5e-3,          # denoise early
    tv_w_final=5e-4,         # small but nonzero later
    warmup_frac=0.1,
    save_intermediate_designs=True,
    save_every=25,
    grad_clip=None,          # e.g., 1.0 to clip global norm
    switch_to_vit_at=None,   # e.g., 1500 -> call model.clip_switch_to_vit()
    clip_every=1,            # >1 to skip CLIP some steps (Adam only)
    clip_ema=0.9             # EMA factor for skipped steps
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    losses = []
    frames = []

    for i in tqdm(range(max_iterations+1), desc="Adam Optimizer"):
        lr      = _cosine_warmup(i, max_iterations, warmup=warmup_frac, start=lr_init,  end=lr_final)
        clip_w  = _cosine_warmup(i, max_iterations, warmup=warmup_frac, start=clip_w_init, end=clip_w_final)
        tv_w    = _cosine_warmup(i, max_iterations, warmup=warmup_frac, start=tv_w_init,   end=tv_w_final)

        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model()
        loss = model.loss(logits, clip_weight=clip_w, tv_weight=tv_w)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach()))
        frames.append(logits.detach().cpu().numpy())
    
    with torch.no_grad():
        designs = [model.env.render(torch.tensor(x), volume_constraint=True) for x in frames]
    return optimizer_result_dataset(np.array(losses), np.array(designs), save_intermediate_designs)

def train_lbfgs(model, max_iterations, save_intermediate_designs=True):
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=max_iterations, line_search_fn='strong_wolfe')
    losses = []
    frames = []

    pbar = tqdm(total=max_iterations, desc="L-BFGS Optimization")

    def closure():
        optimizer.zero_grad()
        logits = model()
        loss = model.loss()
        loss.backward()
        losses.append(loss.item())
        frames.append(logits.detach().cpu().numpy())
        pbar.update(1)
        return loss

    optimizer.step(closure)
    pbar.close()

    designs = [model.env.render(torch.tensor(x), volume_constraint=True) for x in frames]
    return optimizer_result_dataset(np.array(losses), np.array(designs), save_intermediate_designs)

def method_of_moving_asymptotes(
    model, max_iterations, save_intermediate_designs=True, init_model=None
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

  pbar = tqdm(total=max_iterations, desc="MMA Optimization")

  def objective(x):
    return env.objective(x, volume_constraint=False)

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
        pbar.update(1)
      return value
    return wrapper

  losses = []
  frames = []

  opt = nlopt.opt(nlopt.LD_MMA, x0.size)
  opt.set_min_objective(wrap_autograd_func(objective, losses, frames))
  opt.add_inequality_constraint(wrap_autograd_func(constraint))
  opt.set_lower_bounds(1e-2)
  opt.set_upper_bounds(1.0)
  opt.set_maxeval(max_iterations)

  try:
    x = opt.optimize(x0)
    logging.info(f'MMA optimization completed successfully')
  except Exception as e:
    logging.info(f'MMA optimization stopped: {type(e).__name__}: {e}')
  finally:
    pbar.close()

  designs = [env.render(x, volume_constraint=True) for x in frames]
    # Print min/max values across all designs
  designs_array = np.array(designs)
  print(f"Designs min value: {designs_array.min():.4f}")
  print(f"Designs max value: {designs_array.max():.4f}")
  #designs = threshold_projection(np.array(designs))
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

    for i in tqdm(range(max_iterations), desc="Optimality Criteria"):
        try:
            # Apply optimality criteria step
            step_result = topo_physics.optimality_criteria_step(x, env.ke, env.args)
            
            # Handle return value - ensure it's a 1D array
            if isinstance(step_result, tuple):
                x_new = step_result[1]
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
            loss = env.objective(x, volume_constraint=False)
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
            design = env.render(frame, volume_constraint=True)
        except:
            design = frame
        designs.append(design)

    return optimizer_result_dataset(
        np.array(losses), np.array(designs), save_intermediate_designs)

def clip_bootstrap(model, max_iterations, lr=1e-2, tv_w=5e-3, widen=2.0):
    opt = torch.optim.Adam([model.z], lr=lr)
    frames, losses = [], []
    for t in tqdm(range(max_iterations), desc="CLIP Bootstrap"):
        opt.zero_grad(set_to_none=True)
        z = model()  # logits
        # widen sigmoid keeps more pixels in linear region early
        x01 = torch.sigmoid(z / widen)
        Lc  = model.clip_loss(x01).mean().to(z.dtype)
        # Ltv = tv_w * _tv_loss(x01).to(z.dtype) if tv_w > 0 else 0.0
        L   = Lc # + Ltv
        L.backward(); opt.step()

        losses.append(float(Lc.detach()))
        frames.append(torch.sigmoid(model.z.detach()).squeeze().cpu().numpy())

    # return something consistent with your plotting
    return optimizer_result_dataset(np.array(losses), np.array(frames), save_intermediate_designs=True)

# training features

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

# train progressive

def train_progressive(model, max_iterations, resize_num=2, alg=train_adam, save_intermediate_designs=True):
    """Training with tqdm and progressive upsampling. Works on all algs."""

    ds_history = []

    for stage in range(resize_num):
        print(f"\nTraining stage {stage + 1}/{resize_num} at resolution: {model.shape[1]}x{model.shape[2]}")

        ds = alg(model, max_iterations, save_intermediate_designs=save_intermediate_designs)
        ds_history.append(ds)

        # Plot current stage results
        plt.figure(figsize=(8, 4))
        
        # Plot loss curve
        plt.subplot(1, 2, 1)
        loss_df = ds.loss.to_pandas()
        plt.plot(loss_df, linewidth=2)
        plt.title(f'Stage {stage + 1} Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot final design
        plt.subplot(1, 2, 2)
        final_design = ds.design.isel(step=ds.loss.argmin())
        plt.imshow(1 - final_design, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Design ({model.shape[1]}x{model.shape[2]})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        # Upsample after stage if allowed
        if stage < resize_num - 1:
            model.upsample(scale=2)

    # Render all frames
    return ds_history
