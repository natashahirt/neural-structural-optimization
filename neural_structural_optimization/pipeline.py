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

"""
overview:
- launches Apache Beam pipeline that trains all models/distributed optimization across multiple problems
- experiment management, result storage, batch processing
- supports different optimization methods + model configurations
- experiment logging and result aggregation
"""

import ast
import re
import time
import os

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from neural_structural_optimization import models
from neural_structural_optimization.structural import utils as pipeline_utils
from neural_structural_optimization.structural import problems
from neural_structural_optimization.structural import api as topo_api
from neural_structural_optimization.train import (
    Adam_Optimizer, LBFGS_Optimizer, MMA_Optimizer, OptimalityCriteria_Optimizer
)
import numpy as np
import xarray
from apache_beam import runners

# requires tensorflow 2.0


flags.DEFINE_integer(
    'num_seeds', 10,
    'Number of RNG seeds to use.',
)
flags.DEFINE_integer(
    'optimization_steps', 1000,
    'Maximum number of function evaluations.'
)
flags.DEFINE_string(
    'save_dir', 'experiments',
    'Base directory for saving experiment results',
)
flags.DEFINE_string(
    'experiment_name', None,
    'Name of the current experiment.',
)
flags.DEFINE_string(
    'problem_filter', None,
    'If set, only run patterns matching this regex.',
)
flags.DEFINE_string(
    'cnn_kwargs', None,
    'Comma separated list of arguments in the form k=v to pass on to '
    'the CNN model constructor.',
)
flags.DEFINE_boolean(
    'dynamic_depth', False,
    'If true, decide the depth of the CNN dynamically based on problem size.'
)
flags.DEFINE_boolean(
    'quick_run', False,
    'If True, do a quick run (only small optimization problems).',
)
FLAGS = flags.FLAGS


# Quantize saved designs into a single byte per pixel
_BINARY_IMAGE_ENCODING = {
    'dtype': 'int8',
    'scale_factor': 1 / 254,
    'add_offset': 0.5,
}

_TRAIN_FUNCS = {
    'adam': lambda model, **kwargs: Adam_Optimizer(model, **kwargs).optimize(),
    'lbfgs': lambda model, **kwargs: LBFGS_Optimizer(model, **kwargs).optimize(),
    'oc': lambda model, **kwargs: OptimalityCriteria_Optimizer(model, **kwargs).optimize(),
    'mma': lambda model, **kwargs: MMA_Optimizer(model, **kwargs).optimize(),
}


def _get_model_class(name):
  if name == 'pixels':
    return models.PixelModel

  if name == 'cnn':
    return models.CNNModel

  raise ValueError(f'invalid model class: {name}')


def get_beam_counter(name):
  return beam.metrics.Metrics.counter(
      'neural_structural_optimization_pipeline', name)


def save_optimization_task(inputs):
  """Save a single optimization task."""
  problem_name, seed, model_name, optimizer_name = inputs
  problem = problems.PROBLEMS_BY_NAME[problem_name]

  start_time = time.time()
  model = models.MODELS_BY_NAME[model_name](problem)
  
  if optimizer_name in _TRAIN_FUNCS:
    history = _TRAIN_FUNCS[optimizer_name](model, max_iterations=200)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer_name}')
    
  elapsed_time = time.time() - start_time

  # save model
  base_dir = f'{FLAGS.save_dir}/{FLAGS.experiment_name}/{problem.name}/'
  method = f'{model_name}-{optimizer_name}'
  os.makedirs(base_dir, exist_ok=True)
  base_path = f'{base_dir}/{method}_seed{seed}'
  model.save_weights(f'{base_path}.tf_savedmodel')

  # save history
  history.coords.update(dict(
      problem_name=problem.name,
      method=method,
      seed=seed,
      density=problem.density,
      design_area=np.mean(problem.mask),
      width=problem.width,
      height=problem.height,
      elapsed_time=elapsed_time,
  ))
  with open(f'{base_path}.nc', 'wb') as f:
    f.write(history.to_netcdf(encoding={'design': _BINARY_IMAGE_ENCODING}))

  # Ensure design has a step dimension before indexing
  if 'step' not in history.design.dims:
    history = history.expand_dims(step=[0])
    
  best_design = history.design.isel(step=history.loss.argmin(), drop=True)

  # TODO(shoyer): save some GIFs!
  image = pipeline_utils.image_from_design(best_design, problem)
  with open(f'{base_path}.png', 'wb') as f:
    image.save(f, format='png')

  partial_history = history.assign(design=best_design)

  get_beam_counter(f'problems/{problem_name}').inc()
  get_beam_counter(f'method/{method}').inc()

  return (problem_name, method), partial_history


def groupby_seeds(inputs):
  """Group saved optimization tasks by seed."""
  (problem_name, method), histories = inputs
  problem = problems.PROBLEMS_BY_NAME[problem_name]

  history = xarray.concat(histories, dim='seed').sortby('seed')

  base_path = (f'{FLAGS.save_dir}/{FLAGS.experiment_name}/{problem.name}/'
               f'{method}')

  best_seed = int(history.loss.min('step').argmin())
  image = pipeline_utils.image_from_design(
      history.isel(seed=best_seed).design, problem)
  with open(f'{base_path}_best.png', 'wb') as f:
    image.save(f, format='png')

  median_seed = int(
      history.loss.min('step').data.argsort()[history.sizes['seed'] // 2])
  image = pipeline_utils.image_from_design(
      history.isel(seed=median_seed).design, problem)
  with open(f'{base_path}_median.png', 'wb') as f:
    image.save(f, format='png')

  return problem.name, history[['loss']]


def groupby_methods(inputs):
  """Group saved optimization tasks by method."""
  _, histories = inputs
  return xarray.concat(histories, dim='method').sortby('method')


def save_all_losses(histories):
  """Save losses for all tasks into one summary file."""
  history = xarray.concat(histories, dim='problem_name').sortby('problem_name')
  path = f'{FLAGS.save_dir}/{FLAGS.experiment_name}/all_losses.nc'
  with open(path, 'wb') as f:
    f.write(history.to_netcdf())


def main(_, runner=None):
  # must create before flags are used
  if runner is None:
    runner = runners.DirectRunner()

  tasks = []
  for problem in problems.PROBLEMS_BY_NAME.values():
    if (FLAGS.problem_filter
        and not re.search(FLAGS.problem_filter, problem.name)):
      continue

    if FLAGS.quick_run and problem.width * problem.height > 64 ** 2:
      continue

    for seed in range(-1, FLAGS.num_seeds):
      if seed >= 0:
        tasks.append((problem.name, seed, 'cnn', 'lbfgs'))
      tasks.append((problem.name, seed, 'pixels', 'lbfgs'))
      tasks.append((problem.name, seed, 'pixels', 'oc'))
      tasks.append((problem.name, seed, 'pixels', 'mma'))

  if not tasks:
    raise RuntimeError('no tasks to run')

  pipeline = (
      beam.Create(tasks)
      | beam.Map(save_optimization_task)
      | beam.Reshuffle()  # don't fuse optimizations together
      | 'group seeds' >> beam.GroupByKey()
      | beam.Map(groupby_seeds)
      | 'group methods' >> beam.GroupByKey()
      | beam.Map(groupby_methods)
      | beam.combiners.ToList()
      | beam.Map(save_all_losses)
  )
  runner.run(pipeline)


if __name__ == '__main__':
  app.run(main)
