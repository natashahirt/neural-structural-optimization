
from absl import logging
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
from neural_structural_optimization import models  # Add this import
from neural_structural_optimization import topo_physics, topo_api
from neural_structural_optimization.train import *
import scipy.optimize
import tensorflow as tf
import xarray
from tqdm import tqdm

"""
def adaptive_train_lbfgs(
  problem,
  resolutions,
  max_iter_per_stage,
  model_class=models.CNNModelDynamic,
  **kwargs
):
  # incrementally sizes up the latent space
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
"""