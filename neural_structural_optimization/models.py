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
# pylint: disable=invalid-name

import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topo_api, pipeline_utils
from neural_structural_optimization.models_utils import *
from neural_structural_optimization.problems_utils import ProblemParams
import tensorflow as tf
import tensorflow.image as tfi  # for resizing
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# requires tensorflow 2.0

layers = tf.keras.layers

class Model(tf.keras.Model):

  def __init__(self, problem_params=None, seed=None, args=None):
    super().__init__()
    
    if problem_params is not None:
      if isinstance(problem_params, dict):
        self.problem_params = ProblemParams(**problem_params)
      else:
        self.problem_params = problem_params
        
      # Generate args from problem_params if not provided
      if args is None:
          problem = self.problem_params.get_problem()
          args = topo_api.specified_task(problem)
    
    else:
        self.problem_params = None

    set_random_seed(seed)
    self.seed = seed
    self.env = topo_api.Environment(args)
    self.args = args

  # def loss(self, logits):
  #   # for our neural network, we use float32, but we use float64 for the physics
  #   # to avoid any chance of overflow.
  #   # add 0.0 to work-around bug in grad of tf.cast on NumPy arrays
  #   logits = 0.0 + tf.cast(logits, tf.float64)
  #   f = lambda x: batched_topo_loss(x, [self.env])
  #   losses = convert_autograd_to_tensorflow(f)(logits)
  #   return tf.reduce_mean(losses)

  def loss(self, logits):
    logits = tf.cast(logits, tf.float64)
    flat_logits = tf.reshape(logits, [logits.shape[0], -1])  # flatten for autograd

    # Build wrapper on-the-fly to ensure `self.env` is current
    def autograd_loss(x_np):
        return batched_topo_loss(x_np, [self.env])

    wrapped_loss = convert_autograd_to_tensorflow(autograd_loss)
    losses = wrapped_loss(flat_logits)

    return tf.reduce_mean(losses)

  def update_problem_params(self, new_params):
    if isinstance(new_params, dict):
        new_params = ProblemParams(**new_params)
    
    self.problem_params = new_params
    problem = new_params.get_problem()
    new_args = topo_api.specified_task(problem)
    
    # Update environment
    self.env = topo_api.Environment(new_args)
    self.args = new_args

    return new_args

class PixelModel(Model):

  def __init__(self, problem_params=None, seed=None):
    super().__init__(problem_params=problem_params, seed=seed)
    shape = (1, self.env.args['nely'], self.env.args['nelx'])
    z_init = np.broadcast_to(self.env.args['volfrac'] * self.env.args['mask'], shape)
    self.z = tf.Variable(z_init, trainable=True)

  def call(self, inputs=None):
    return self.z

class PixelModelAdaptive(Model):

  def __init__(self, 
               problem_params=None, 
               resize_num=0, 
               resize_scale=2, 
               args=None, 
               seed=None
  ):
    super().__init__(problem_params=problem_params, seed=seed, args=args)
    
    self.resize_num = resize_num
    self.resize_scale = resize_scale
    self.resizes = 0
    self.problem_params = problem_params
    
    self.shape = (1, self.env.args['nely'], self.env.args['nelx'])
    z_init = np.broadcast_to(self.env.args['volfrac'] * self.env.args['mask'], self.shape)
    self.z = tf.Variable(z_init, trainable=True, dtype=tf.float32)

    self.prev_loss = tf.constant(1e5, dtype=tf.float32)

  def call(self, inputs=None):
    return self.z

  def threshold_crossed(self, loss, threshold=0.05):
    delta = tf.abs(self.prev_loss - loss)
    # self.prev_loss = loss # without this it seems kind of useless
    return delta < threshold

  def upsample(self):
    if self.problem_params is None:
      raise ValueError("No problem_params available for scaling")
    
    # update problem params
    new_problem_params = self.problem_params.copy(
                width=int(self.problem_params.width * self.resize_scale),
                height=int(self.problem_params.height * self.resize_scale)
            )
    new_problem_params.interval = int(self.problem_params.interval * self.resize_scale)
    if self.problem_params.num_points > 1:
      # Scale num_points but keep it reasonable (don't scale if it's already large)
      scaled_points = int(self.problem_params.num_points * self.resize_scale)
      new_problem_params.num_points = min(scaled_points, 50)  # Cap at reasonable maximum

    self.update_problem_params(new_problem_params)

    self.shape = (1, self.env.args['nely'], self.env.args['nelx'])
    
    # resize z
    z_resized = tfi.resize(self.z, 
      size=[int(self.shape[1] * self.resize_scale),
            int(self.shape[2] * self.resize_scale)],
      method="bilinear", antialias=True)

    z_resized = tf.clip_by_value(z_resized, 0.0, 1.0)
    z_resized = tf.expand_dims(z_resized[0], axis=0)  # ensure shape (1, H, W)
    
    self.z = tf.Variable(z_resized, trainable=True, dtype=tf.float32)
    self.resizes += 1

class CNNModel(Model):

  def __init__(
      self,
      seed=0,
      args=None,
      problem_params=None,
      latent_size=128,
      dense_channels=32,
      resizes=(1, 2, 2, 2, 1),
      conv_filters=(128, 64, 32, 16, 1),
      offset_scale=10,
      kernel_size=(5, 5),
      latent_scale=1.0,
      dense_init_scale=1.0,
      activation=tf.nn.tanh,
      conv_initializer=tf.initializers.VarianceScaling,
      normalization=global_normalization,
  ):
    super().__init__(problem_params=problem_params, seed=seed, args=args)

    if len(resizes) != len(conv_filters):
      raise ValueError('resizes and filters must be same size')

    activation = layers.Activation(activation)

    total_resize = int(np.prod(resizes))
    h = self.env.args['nely'] // total_resize
    w = self.env.args['nelx'] // total_resize

    net = inputs = layers.Input((latent_size,), batch_size=1)
    filters = h * w * dense_channels
    dense_initializer = tf.initializers.orthogonal(
        dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
    net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
    net = layers.Reshape([h, w, dense_channels])(net)

    for resize, filters in zip(resizes, conv_filters):
      net = activation(net)
      net = UpSampling2D(resize)(net)
      net = normalization(net)
      net = Conv2D(
          filters, kernel_size, kernel_initializer=conv_initializer)(net)
      if offset_scale != 0:
        net = AddOffset(offset_scale)(net)

    outputs = tf.squeeze(net, axis=[-1])

    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z')

  def call(self, inputs=None):
    return self.core_model(self.z)