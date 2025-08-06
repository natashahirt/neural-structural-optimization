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

import numpy as np

import tensorflow as tf
import tensorflow.image as tfi
from scipy.ndimage import zoom

from neural_structural_optimization.models_utils import (
    set_random_seed, batched_topo_loss, convert_autograd_to_tensorflow,
    global_normalization, UpSampling2D, Conv2D, AddOffset
)
from neural_structural_optimization.clip_config import CLIPConfig
from neural_structural_optimization.clip_loss import CLIPLoss
from neural_structural_optimization.problems_utils import ProblemParams
from neural_structural_optimization.pipeline_utils import dynamic_depth_kwargs
from neural_structural_optimization import topo_api

# requires tensorflow 2.0
layers = tf.keras.layers

class Model(tf.keras.Model):

  def __init__(self, problem_params=None, clip_config=None, seed=None, args=None):
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

    #initialize losses
    self.clip_loss = None
    self.clip_weight = 0.0
    if clip_config is not None and clip_config.weight > 0:
      self.clip_loss = CLIPLoss(config=clip_config)
      self.clip_weight = clip_config.weight

  # def loss(self, logits):
  #   # for our neural network, we use float32, but we use float64 for the physics
  #   # to avoid any chance of overflow.
  #   # add 0.0 to work-around bug in grad of tf.cast on NumPy arrays
  #   logits = 0.0 + tf.cast(logits, tf.float64)
  #   f = lambda x: batched_topo_loss(x, [self.env])
  #   physics_loss = convert_autograd_to_tensorflow(f)(logits)
  #   return tf.reduce_mean(physics_loss)

  def loss(self, logits):
    # logits = 0.0 + tf.cast(logits, tf.float64)
    # f = lambda x: batched_topo_loss(x, [self.env])
    # physics_loss = convert_autograd_to_tensorflow(f)(logits)
    # physics_loss = tf.reduce_mean(physics_loss)

    clip_loss = tf.constant(0.0, dtype=tf.float64)
    if self.clip_loss is not None:
        if self.clip_loss.target_text_prompt is not None: 
          clip_loss = self.clip_loss.get_text_loss(logits, self.clip_loss.target_text_prompt)
          clip_loss = tf.cast(clip_loss, tf.float64)

    clip_loss = tf.reduce_mean(clip_loss)

    return clip_loss

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

  def __init__(self, problem_params=None, clip_config=None, seed=None):
    super().__init__(problem_params=problem_params, clip_config=clip_config, seed=seed)
    self.shape = (1, self.env.args['nely'], self.env.args['nelx'])
    z_init = np.broadcast_to(self.env.args['volfrac'] * self.env.args['mask'], self.shape)
    self.z = tf.Variable(z_init, trainable=True, dtype=tf.float32)

  def call(self, inputs=None):
    return self.z

class PixelModelAdaptive(PixelModel):

  def __init__(self, 
               problem_params=None, 
               clip_config=None,
               resize_num=2, 
               resize_scale=2, 
               args=None, 
               seed=None
  ):
    super().__init__(problem_params=problem_params, clip_config=clip_config, seed=seed)
    
    self.resize_num = resize_num
    self.resize_scale = resize_scale
    self.problem_params = problem_params
    self.prev_loss = tf.constant(1e5, dtype=tf.float32)

  def threshold_crossed(self, loss, threshold=0.05):
    delta = tf.abs(self.prev_loss - loss)
    # self.prev_loss = loss # without this it seems kind of useless
    return delta < threshold

  def upsample(self):
    if self.problem_params is None:
      raise ValueError("No problem_params available for scaling")
    
    # update problem params
    scaled_width=int(self.problem_params.width * self.resize_scale)
    scaled_height=int(self.problem_params.height * self.resize_scale)
    
    new_problem_params = self.problem_params.copy(width=scaled_width, height=scaled_height)
    self.update_problem_params(new_problem_params)

    self.shape = (1, new_problem_params.height, new_problem_params.width)

    # self.upsample_z_tf()
    self.upsample_z_scipy_zoom()
    
  # helper methods for upsampling

  def upsample_z_tf(self):
    # straight upsample
    z = self.z

    z = tf.expand_dims(z, axis=-1) # add extra channel dimension for tfi

    # resize z
    z_resized = tfi.resize(z, 
      size=[int(self.shape[1]), int(self.shape[2])],
      method="bilinear", antialias=True)

    z_resized = tf.squeeze(z_resized, axis=-1) # get rid of extra channel dimension
    z_resized = tf.clip_by_value(z_resized, 0.0, 1.0)
    z_resized = tf.expand_dims(z_resized[0], axis=0)  # ensure shape (1, H, W)
    tf.keras.backend.clear_session()

    self.z = tf.Variable(z_resized, trainable=True, dtype=tf.float32)
  
  def upsample_z_scipy_zoom(self):
    # chunked upsample

    z = self.z

    z_np = z.numpy()
    z_expanded = np.expand_dims(z_np, axis=-1)
    zoom_factors = (1, self.resize_scale, self.resize_scale, 1)
    z_resized = zoom(z_expanded, zoom_factors, order=1)
    z_resized = np.squeeze(z_resized, axis=-1)
    z_resized = np.clip(z_resized, 0.0, 1.0)
    if z_resized.shape != self.shape:
      z_resized = np.expand_dims(z_resized[0], axis=0)
    
    tf.keras.backend.clear_session()
    
    self.z = tf.Variable(z_resized, trainable=True, dtype=tf.float32)

class CNNModel(Model):

  def __init__(
      self,
      problem_params=None,
      clip_config=None,
      latent_size=128,
      dense_channels=32,
      upsample_factors=(1, 2, 2, 2, 1),
      conv_filters=(128, 64, 32, 16, 1),
      offset_scale=10,
      kernel_size=(5, 5),
      activation=tf.nn.tanh,
      conv_initializer=tf.initializers.VarianceScaling,
      normalization=global_normalization,
  ):
    super().__init__(problem_params=problem_params, clip_config=clip_config)

    if len(upsample_factors) != len(conv_filters):
      raise ValueError('upsample_factors and filters must be same size')

    activation = layers.Activation(activation)

    total_upsample = int(np.prod(upsample_factors))
    h = self.env.args['nely'] // total_upsample
    w = self.env.args['nelx'] // total_upsample

    net = inputs = layers.Input((latent_size,), batch_size=1)
    filters = h * w * dense_channels
    dense_initializer = tf.initializers.orthogonal(
        1.0 * np.sqrt(max(filters / latent_size, 1)))
    net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
    net = layers.Reshape([h, w, dense_channels])(net)

    for upsample_factor, filters in zip(upsample_factors, conv_filters):
      net = activation(net)
      net = UpSampling2D(upsample_factor)(net)
      net = normalization(net)
      net = Conv2D(
          filters, kernel_size, kernel_initializer=conv_initializer)(net)
      if offset_scale != 0:
        net = AddOffset(offset_scale)(net)

    outputs = tf.squeeze(net, axis=[-1])

    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=1.0)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z')
  
  @property
  def shape(self):
    return (1, self.env.args['nely'], self.env.args['nelx'])

  def call(self, inputs=None):
    return self.model(self.z)

class CNNModelAdaptive(CNNModel):

  def __init__(self,
               problem_params=None,
               clip_config=None,
               resize_num=2, 
               resize_scale=2,
               latent_size=128,
               dense_channels=32,
               upsample_factors=(1, 2, 2, 2, 1),
               conv_filters=(128, 64, 32, 16, 1),
               offset_scale=10,
               kernel_size=(5, 5),
               activation=tf.nn.tanh,
               conv_initializer=tf.initializers.VarianceScaling,
               normalization=global_normalization,
  ):
    # Store parameters for rebuilding
    self.latent_size = latent_size
    self.dense_channels = dense_channels
    self.upsample_factors = upsample_factors
    self.conv_filters = conv_filters
    self.offset_scale = offset_scale
    self.kernel_size = kernel_size
    self.activation = activation
    self.conv_initializer = conv_initializer
    self.normalization = normalization
    
    # Initialize parent with current parameters
    super().__init__(
        problem_params=problem_params,
        clip_config=clip_config,
        latent_size=latent_size,
        dense_channels=dense_channels,
        upsample_factors=upsample_factors,
        conv_filters=conv_filters,
        offset_scale=offset_scale,
        activation=activation,
        kernel_size=kernel_size,
        conv_initializer=conv_initializer,
        normalization=normalization
    )

    self.resize_scale = resize_scale
    self.resize_num = resize_num
    self.prev_loss = None

  def threshold_crossed(self, loss, threshold=0.05):
    if self.prev_loss is None:
      self.prev_loss = loss
      return False
    delta = tf.abs(self.prev_loss - loss)
    # self.prev_loss = loss  # uncomment to update during training
    return delta < threshold

  def upsample(self):
    if self.problem_params is None:
      raise ValueError("No problem_params available for scaling")

    # Update problem dimensions
    new_width = int(self.problem_params.width * self.resize_scale)
    new_height = int(self.problem_params.height * self.resize_scale)
    new_problem_params = self.problem_params.copy(width=new_width, height=new_height)

    new_problem_params, cnn_kwargs = dynamic_depth_kwargs(new_problem_params)

    self.update_problem_params(new_problem_params)

    self.upsample_factors = cnn_kwargs['upsample_factors']
    self.conv_filters = cnn_kwargs['conv_filters']
    self.dense_channels = cnn_kwargs['dense_channels']

    # Get weights from current model before rebuilding
    old_weights = self.z.numpy()

    # Build new model with greater upsamplings / larger dense layer
    self._rebuild_model()

  def _rebuild_model(self):
    # get old weights
    prev_weights = {layer.name: [w.numpy() for w in layer.weights] for layer in self.model.layers}

    activation = layers.Activation(tf.nn.tanh) 
    
    total_upsample = int(np.prod(self.upsample_factors))
    h = int(self.problem_params.height // total_upsample)
    w = int(self.problem_params.width // total_upsample)

    # input layer
    net = inputs = layers.Input((self.latent_size,), batch_size=1)
    filters = h * w * self.dense_channels

    # dense layer initialization
    dense_initializer = tf.initializers.orthogonal(
        np.sqrt(max(filters / self.latent_size, 1)))
    net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
    net = layers.Reshape([h, w, self.dense_channels])(net)
    
    for i, (upsample_factor, filters) in enumerate(zip(self.upsample_factors, self.conv_filters)):
        idx = len(self.upsample_factors) - i - 1
        
        # apply activation
        net = activation(net)

        net = layers.UpSampling2D(upsample_factor, name=f"upsampling_{idx}")(net)
        net = self.normalization(net)
        net = layers.Conv2D(
            filters, 
            self.kernel_size, 
            kernel_initializer=self.conv_initializer,
            padding='same',
            name=f"conv2d_{idx}"
        )(net)

        if i > 0 and filters == self.conv_filters[i-1]:
            # Add residual connection if filter counts match
            net = layers.Add(name=f"residual_{idx}")([net, residual_input])
        
        # Store for potential residual connection
        residual_input = net

        if self.offset_scale != 0:
            net = AddOffset(self.offset_scale)(net)

    outputs = tf.squeeze(net, axis=[-1])
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    transferred = 0
    for layer in self.model.layers:
      if layer.name in prev_weights and layer.weights:
        try:
          # Check if weight shapes are compatible
          prev_weight_shapes = [w.shape for w in prev_weights[layer.name]]
          new_weight_shapes = [w.shape for w in layer.weights]
          
          if prev_weight_shapes == new_weight_shapes:
              layer.set_weights(prev_weights[layer.name])
              layer.trainable = False
              transferred += 1
              print(f"Transferred and froze weights for layer: {layer.name}")
          else:
              print(f"Skipped {layer.name}: shape mismatch {prev_weight_shapes} vs {new_weight_shapes}")
        except Exception as e:
            print(f"Failed to transfer {layer.name}: {e}")
            continue

    print(f"Transferred weights for {transferred} layers")