"""API interface for structural optimization.

This module provides a clean interface between physics engine and neural networks,
handling problem parameter setup, boundary conditions, forces, and constraints.
"""

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

"""
overview:
- clean interface between physics engine and neural networks
- problem parameter setup (boundary conditions, forces, constraints)
- environment class wraps physics calcs
- converts between different coordinate systems and data formats
"""

import autograd.numpy as np
from neural_structural_optimization.structural import physics
from typing import Any, Dict

import numpy as _np
try:
    import torch as _torch
except Exception:  # torch not required here, but supported
    _torch = None

def _to_numpy(x, dtype=_np.float64):
    """Convert torch.Tensor / list / np.array to np.ndarray(dtype), no copy if possible."""
    if _torch is not None and _torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
  
    if hasattr(x, "values"): # xarray.DataArray / Dataset-like
        return np.asarray(x.values, dtype=dtype)

    if isinstance(x, np.ndarray) and x.dtype != object: # already a NumPy array of numeric dtype:
        return x.astype(dtype, copy=False)

    if isinstance(x, (list, tuple)) and len(x) == 1: # 1-element lists/tuples that wrap an array
        return _to_numpy(x[0], dtype=dtype)

    arr = np.array(x, dtype=dtype) # try array() and detect object dtype
    if arr.dtype == object:
        raise TypeError(f"Expected a numeric array/tensor; got sequence of objects: {type(x)}")
    return arr

def _args_to_numpy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure physics args are NumPy-native (arrays/scalars)."""
    out = dict(args)

    # arrays we expect to be array-like
    for key in ("mask", "freedofs", "fixdofs", "forces"):
        if key in out:
            # freedofs/fixdofs should be int64
            if key in ("freedofs", "fixdofs"):
                arr = _to_numpy(out[key], dtype=_np.int64).ravel()
                out[key] = arr
            elif key == "forces":
                out[key] = _to_numpy(out[key], dtype=_np.float64).ravel()
            else:  # mask
                out[key] = _to_numpy(out[key], dtype=_np.float64)

    # scalar-ish fields: make sure they are plain Python/NumPy scalars
    for key in ("young", "young_min", "poisson", "g",
                "volfrac", "xmin", "xmax", "nelx", "nely",
                "penal", "filter_width"):
        if key in out:
            v = out[key]
            if _torch is not None and _torch.is_tensor(v):
                v = v.detach().cpu().item()
            elif hasattr(v, "item") and not isinstance(v, (bytes, str)):
                try:
                    v = v.item()
                except Exception:
                    pass
            out[key] = v

    return out

def specified_task(problem):
    """Given a problem, return parameters for running topology optimization (NumPy)."""
    normals = _to_numpy(problem.normals, dtype=_np.float64)
    forces  = _to_numpy(problem.forces,  dtype=_np.float64)
    mask    = _to_numpy(problem.mask,    dtype=_np.float64) if not _np.isscalar(problem.mask) else problem.mask

    # fixed dofs from normals==1 (the original code flattened the whole 3D array)
    fixdofs = _np.flatnonzero(normals.ravel())
    alldofs = _np.arange(2 * (problem.width + 1) * (problem.height + 1), dtype=_np.int64)
    freedofs = _np.sort(_np.setdiff1d(alldofs, fixdofs, assume_unique=False))

    params = {
        # material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # constraints
        "volfrac": float(problem.density),
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": int(problem.width),
        "nely": int(problem.height),
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces.ravel(),
        "penal": 3.0,
        "filter_width": 2,
    }
    return params

class Environment:
    """NumPy-only wrapper around physics. Accepts torch/np at the boundary."""

    def __init__(self, args):
        # normalize args to NumPy once
        self.args = _args_to_numpy(args)
        # stiffness matrix (NumPy)
        self.ke = physics.get_stiffness_matrix(self.args["young"], self.args["poisson"])

    def reshape(self, params):
        p = _to_numpy(params)  # ensure NumPy
        return p.reshape(self.args["nely"], self.args["nelx"])

    def render(self, params, volume_constraint=True):
        x2d = self.reshape(params)
        return physics.physical_density(
            x2d, self.args, volume_constraint=volume_constraint, cone_filter=False
        )

    def objective(self, params, volume_constraint=False):
        x2d = self.reshape(params)
        return physics.objective(
            x2d, self.ke, self.args, volume_constraint=volume_constraint, cone_filter=True
        )

    def constraint(self, params):
        x2d = self.reshape(params)
        vol = physics.mean_density(x2d, self.args)
        return vol - self.args["volfrac"]