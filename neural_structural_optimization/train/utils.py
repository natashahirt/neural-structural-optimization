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

"""Utility functions for optimization framework."""

import numpy as np
import torch
from absl import logging

from neural_structural_optimization.structural import physics


def get_variables(model) -> np.ndarray:
    """Get flattened array from PyTorch model parameters."""
    return np.concatenate([
        v.detach().cpu().numpy().ravel() 
        for v in model.parameters() if v.requires_grad])

def constrained_logits(init_model) -> np.ndarray:
    """Produce matching initial conditions with volume constraints applied."""
    logits = init_model().detach().cpu().numpy().astype(np.float64).squeeze(axis=0)
    return physics.physical_density(
        logits, init_model.env.args, volume_constraint=True, cone_filter=False)

def cosine_warmup(t: int, T: int, warmup: float = 0.1, start: float = 1.0, end: float = 0.0) -> float:
    """Cosine from `start`→`end` after a linear warmup portion."""
    Tw = max(int(T * warmup), 1)
    if t < Tw:
        return start * (t + 1) / Tw
    tt = (t - Tw) / max(T - Tw, 1)
    return end + 0.5 * (start - end) * (1 + np.cos(np.pi * tt))

def ensure_array_size(x: np.ndarray, expected_size: int, name: str = "array") -> np.ndarray:
    """Ensure array has correct size with appropriate padding or truncation."""
    x = np.asarray(x).ravel()
    if x.size != expected_size:
        logging.warning(f'Reshaping {name} from {x.size} to {expected_size}')
        if x.size == 1:
            x = np.full(expected_size, float(x[0]))
        else:
            x = x[:expected_size]
            if x.size < expected_size:
                x = np.pad(x, (0, expected_size - x.size), mode='edge')
    return x
