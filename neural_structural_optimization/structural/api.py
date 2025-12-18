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
- Environment class wraps physics calcs
- This implementation prefers the pyFANTOM backend; falls back to legacy autograd physics if unavailable
"""

from typing import Any, Dict, Tuple
import autograd.numpy as np
import numpy as _np

# Optional torch boundary conversions
try:
  import torch as _torch  # type: ignore
except Exception:
  _torch = None

# Prefer pyFANTOM if available
_HAS_PYFANTOM = False
try:
  import os
  import sys
  # Add pyFANTOM to path if it's in the workspace but not installed
  _pyfantom_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pyFANTOM")
  if os.path.exists(_pyfantom_path) and _pyfantom_path not in sys.path:
      sys.path.insert(0, _pyfantom_path)

  # Narrow imports to CPU-only to keep import time low
  from pyFANTOM.CPU import (
      StructuredMesh2D,
      StructuredStiffnessKernel,
      FiniteElement,
      StructuredFilter2D,
      CHOLMOD,
      CG,
  )
  from pyFANTOM.CPU import MinimumCompliance  # problem
  from pyFANTOM import LinearElasticity
  _HAS_PYFANTOM = True
  print("Structural backend: pyFANTOM (CPU)")
except Exception as e:
  _HAS_PYFANTOM = False
  import traceback
  print(f"Structural backend: Legacy (autograd)")
  print(f"pyFANTOM import failed. Error: {e}")
  # Optional: print(traceback.format_exc()) # if we need more detail

# Legacy backend (autograd-based)
from neural_structural_optimization.structural import physics

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

    # fixed dofs from normals==1 (flatten the whole 3D array of node DOFs)
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
        "heavyside": getattr(problem, "heavyside", True),
        "beta": getattr(problem, "beta", 2.0),
        "eta": getattr(problem, "eta", 0.5),
    }
    return params

class Environment:
  """Backend wrapper for structural physics.

  Prefers pyFANTOM for FEA/topopt (fast, feature-rich). Falls back to the legacy
  autograd-based physics when pyFANTOM is unavailable.
  """

  def __init__(self, args: Dict[str, Any]):
    # normalize args to NumPy once
    self.args = _args_to_numpy(args)

    # Common shape helpers
    self.nelx: int = int(self.args["nelx"])
    self.nely: int = int(self.args["nely"])
    self._num_elems: int = self.nelx * self.nely

    # Normalize mask as 2D array of shape (nely, nelx)
    _mask = self.args.get("mask", 1.0)
    m = _to_numpy(_mask, dtype=_np.float64)
    if m.size == 1:
      self.mask2d = _np.ones((self.nely, self.nelx), dtype=_np.float64) * float(m.item())
    else:
      # mask provided as array; ensure expected (nely, nelx)
      self.mask2d = m.reshape(self.nely, self.nelx)
    self._mask_flat = self.mask2d.ravel()

    # Backend selection
    self._backend = "pyfantom" if _HAS_PYFANTOM else "legacy"

    if self._backend == "pyfantom":
      self._init_pyfantom_backend()
    else:
      self._init_legacy_backend()

  # ----------------------- Backend initializers -----------------------
  def _init_legacy_backend(self) -> None:
    # Legacy stiffness matrix (NumPy)
    self.ke = physics.get_stiffness_matrix(self.args["young"], self.args["poisson"])

  def _init_pyfantom_backend(self) -> None:
    # pyFANTOM: build a structured 2D mesh over unit square
    E = float(self.args.get("young", 1.0))
    nu = float(self.args.get("poisson", 0.3))
    physics_model = LinearElasticity(E=E, nu=nu, type="PlaneStress")

    self._mesh = StructuredMesh2D(nx=self.nelx, ny=self.nely, lx=1.0, ly=1.0, physics=physics_model)
    self.ke = self._mesh.K_single  # keep compatibility for any callers expecting ke

    self._kernel = StructuredStiffnessKernel(mesh=self._mesh)

    # Choose solver: prefer CHOLMOD if available, else CG
    try:
      self._solver = CHOLMOD(kernel=self._kernel)
    except Exception:
      self._solver = CG(kernel=self._kernel)

    self._fe = FiniteElement(mesh=self._mesh, kernel=self._kernel, solver=self._solver)

    # Apply boundary conditions and forces from args
    self._apply_bc_and_loads()

    # Density filter
    rmin = float(self.args.get("filter_width", 2))
    self._filter = StructuredFilter2D(mesh=self._mesh, r_min=rmin)

    # Problem: Minimum compliance with SIMP
    penal = float(self.args.get("penal", 3.0))
    volfrac = float(self.args.get("volfrac", 0.5))
    void = float(self.args.get("young_min", 1e-9))

    # Configure single-material problem
    self._problem = MinimumCompliance(
        FE=self._fe,
        filter=self._filter,
        E_mul=[1.0],               # material multiplier
        void=void,
        penalty=penal,
        volume_fraction=[volfrac],
        penalty_schedule=None,
        heavyside=bool(self.args.get("heavyside", True)),
        beta=float(self.args.get("beta", 2.0)),
        eta=float(self.args.get("eta", 0.5)),
    )
    # Initialize internal state
    self._problem.init_desvars()

  def _heaviside(self, rho, beta=None, eta=None):
    """Original pyFANTOM tanh-based Heaviside projector."""
    if beta is None:
      beta = float(self.args.get("beta", 2.0))
    if eta is None:
      eta = float(self.args.get("eta", 0.5))
    
    # Formula from pyFANTOM.Problem.CPU.MinimumCompliance.penalize
    num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den

  def _apply_bc_and_loads(self) -> None:
    """Map `fixdofs` and `forces` from our args into pyFANTOM FE object."""
    # fixdofs are global DOF indices (interleaved ux,uy per node). Convert to node-wise masks.
    fixdofs = _to_numpy(self.args.get("fixdofs", _np.array([], dtype=_np.int64)), dtype=_np.int64).ravel()

    if fixdofs.size > 0:
      node_ids = (fixdofs // 2).astype(_np.int64)
      dof_ids = (fixdofs % 2).astype(_np.int64)  # 0 -> ux, 1 -> uy
      # Build per-node DOF mask
      uniq_nodes, inverse = _np.unique(node_ids, return_inverse=True)
      dof_mask = _np.zeros((uniq_nodes.shape[0], 2), dtype=_np.int64)
      dof_mask[inverse, dof_ids] = 1
      self._fe.add_dirichlet_boundary_condition(node_ids=uniq_nodes, dofs=dof_mask, rhs=0.0)

    # Forces: flattened length 2 * (nely+1)*(nelx+1); map to node vectors
    forces_flat = _to_numpy(self.args.get("forces", _np.zeros(2*(self.nelx+1)*(self.nely+1))), dtype=_np.float64)
    num_nodes = (self.nelx + 1) * (self.nely + 1)
    if forces_flat.size == 2 * num_nodes:
      forces_2d = forces_flat.reshape(num_nodes, 2)
      nz_idx = _np.flatnonzero(_np.any(_np.abs(forces_2d) > 0, axis=1))
      if nz_idx.size > 0:
        self._fe.add_point_forces(node_ids=nz_idx, forces=forces_2d[nz_idx])

  # ---------------------------- Utilities -----------------------------
  def reshape(self, params):
    p = _to_numpy(params)  # ensure NumPy
    return p.reshape(self.nely, self.nelx)

  # ----------------------------- Renders ------------------------------
  def render(self, params, volume_constraint=True):
    x2d = self.reshape(params)
    if self._backend == "pyfantom":
      if volume_constraint:
        # Use pyFANTOM projection logic: Filter -> Heaviside
        rho = self._project_volume(x2d.ravel()).reshape(self.nely, self.nelx)
        return rho
      else:
        # No volume constraint: apply mask and filter
        rho = (x2d * self.mask2d).ravel()
        rf = self._filter.dot(rho).reshape(self.nely, self.nelx)
        if self.args.get("heavyside", True):
          rf = self._heaviside(rf)
        return rf * self.mask2d
    
    # Legacy physics
    return physics.physical_density(
        x2d, self.args, volume_constraint=volume_constraint, cone_filter=False
    )

  # ----------------------------- Objective ---------------------------
  def _project_volume(self, x_flat: _np.ndarray) -> _np.ndarray:
    """Find a shift `b` such that mean(H(Filter(sigmoid(x + b)))) = volfrac."""
    volfrac = float(self.args["volfrac"])
    mask = self.mask2d.ravel() > 0
    
    # Define the full projection chain for autograd
    def get_projected_rho(x_in, shift):
      # x_in are raw logits
      rho_raw = physics.sigmoid(x_in + shift)
      
      # Apply design mask
      from neural_structural_optimization.structural import autograd as topo_autograd
      rho_masked = topo_autograd.scatter1d(rho_raw[mask], _np.flatnonzero(mask), x_in.size)
      
      if self._backend == "pyfantom":
        # pyFANTOM order: Filter -> Heaviside
        # Note: self._filter.dot is not autograd-aware by default, 
        # but structured filter is a convolution which we can wrap.
        # Actually, pyFANTOM filters are sparse matrix-vec.
        # We can implement a simple autograd-compatible filter if needed, 
        # or use the one from topo_autograd.
        rf = topo_autograd.cone_filter(rho_masked.reshape(self.nely, self.nelx), 
                                       self.args["filter_width"], 
                                       self.mask2d).ravel()
        if self.args.get("heavyside", True):
          return self._heaviside(rf)
        return rf
      else:
        # Legacy order: Heaviside (sigmoid) -> Filter
        # Sigmoid was already applied in rho_raw.
        rf = topo_autograd.cone_filter(rho_masked.reshape(self.nely, self.nelx), 
                                       self.args["filter_width"], 
                                       self.mask2d).ravel()
        return rf

    def f_root(x_in, shift):
      return np.mean(get_projected_rho(x_in, shift)) - volfrac

    # Find shift using root finder
    import autograd
    from neural_structural_optimization.structural import autograd as topo_autograd
    
    # Bounds for the shift b
    lower = physics.logit(volfrac) - np.max(x_flat[mask])
    upper = physics.logit(volfrac) - np.min(x_flat[mask])
    
    b = topo_autograd.find_root(f_root, x_flat, lower, upper)
    return get_projected_rho(x_flat, b)

  def objective(self, params, volume_constraint=False):
    if self._backend == "pyfantom":
      x_flat = _to_numpy(params, dtype=_np.float64).ravel()
      if volume_constraint:
        # Find design variables that satisfy the physical volume constraint
        rho_design = self._project_volume_design(x_flat)
      else:
        # standard SIMP path: design variables are masked sigmoid of logits
        mask = self.mask2d.ravel() > 0
        rho_design = _np.zeros_like(x_flat)
        rho_design[mask] = physics.sigmoid(x_flat[mask])
      
      # pyFANTOM's set_desvars will apply Filter -> Heaviside -> SIMP internally
      self._problem.set_desvars(rho_design.copy())
      return float(self._problem.f())
    
    # Legacy physics
    x2d = self.reshape(params)
    return physics.objective(
        x2d, self.ke, self.args, volume_constraint=volume_constraint, cone_filter=True
    )

  def objective_grad(self, params, volume_constraint=False):
    if self._backend != "pyfantom":
      raise RuntimeError("objective_grad is only available with the pyFANTOM backend")

    x_flat = _to_numpy(params, dtype=_np.float64).ravel()
    
    if not volume_constraint:
      # Simple chain rule through sigmoid and mask
      mask = self.mask2d.ravel() > 0
      rho_design = _np.zeros_like(x_flat)
      rho_design[mask] = physics.sigmoid(x_flat[mask])
      
      self._problem.set_desvars(rho_design.copy())
      dC_drho_design = self._problem.nabla_f() # dC/drho_design from pyFANTOM
      
      # dC/dlogits = dC/drho_design * drho_design/dlogits
      import autograd
      vjp_sigmoid = autograd.make_vjp(physics.sigmoid)(x_flat[mask])[1]
      g_design = vjp_sigmoid(dC_drho_design[mask])
      
      g_out = _np.zeros_like(x_flat)
      g_out[mask] = g_design
      return g_out.reshape(self.nely, self.nelx)

    # With volume constraint: chain rule through projection
    import autograd
    from neural_structural_optimization.structural import autograd as topo_autograd
    
    # We need the VJP of the function that maps logits to design variables rho_design
    # such that PhysicalVolume(rho_design) = volfrac.
    def get_rho_design_from_logits(x_in):
      volfrac = float(self.args["volfrac"])
      mask = self.mask2d.ravel() > 0
      
      def f_root(x_slice, shift):
        # Physical volume as function of shift
        # pyFANTOM order: H(Filter(rho_design))
        rho_d = physics.sigmoid(x_slice + shift)
        rho_m = topo_autograd.scatter1d(rho_d, _np.flatnonzero(mask), x_in.size)
        rf = topo_autograd.cone_filter(rho_m.reshape(self.nely, self.nelx), 
                                       self.args["filter_width"], 
                                       self.mask2d).ravel()
        if self.args.get("heavyside", True):
          proj = self._heaviside(rf)
        else:
          proj = rf
        return np.mean(proj) - volfrac

      # Find shift b(logits)
      lower = physics.logit(volfrac) - np.max(x_in[mask])
      upper = physics.logit(volfrac) - np.min(x_in[mask])
      b = topo_autograd.find_root(f_root, x_in[mask], lower, upper)
      
      # Final design variables passed to pyFANTOM
      rho_d_final = physics.sigmoid(x_in[mask] + b)
      return topo_autograd.scatter1d(rho_d_final, _np.flatnonzero(mask), x_in.size)

    # 1. Forward pass to get dC/drho_design from pyFANTOM
    rho_design = get_rho_design_from_logits(x_flat)
    self._problem.set_desvars(rho_design.copy())
    dC_drho_design = self._problem.nabla_f()
    
    # 2. Backward pass using VJP
    vjp_func = autograd.make_vjp(get_rho_design_from_logits)(x_flat)[1]
    g_raw = vjp_func(dC_drho_design)
    
    return g_raw.reshape(self.nely, self.nelx)

  def _project_volume_design(self, x_flat: _np.ndarray) -> _np.ndarray:
    """Find design variables rho_design such that mean(PhysicalVolume(rho_design)) = volfrac."""
    volfrac = float(self.args["volfrac"])
    mask = self.mask2d.ravel() > 0
    
    from neural_structural_optimization.structural import autograd as topo_autograd
    
    def f_root(x_slice, shift):
      rho_d = physics.sigmoid(x_slice + shift)
      rho_m = topo_autograd.scatter1d(rho_d, _np.flatnonzero(mask), x_flat.size)
      rf = topo_autograd.cone_filter(rho_m.reshape(self.nely, self.nelx), 
                                     self.args["filter_width"], 
                                     self.mask2d).ravel()
      if self.args.get("heavyside", True):
        proj = self._heaviside(rf)
      else:
        proj = rf
      return np.mean(proj) - volfrac

    lower = physics.logit(volfrac) - np.max(x_flat[mask])
    upper = physics.logit(volfrac) - np.min(x_flat[mask])
    b = topo_autograd.find_root(f_root, x_flat[mask], lower, upper)
    
    rho_d_final = physics.sigmoid(x_flat[mask] + b)
    out = _np.zeros_like(x_flat)
    out[mask] = rho_d_final
    return out

  def _masked_flat(self, params: _np.ndarray) -> _np.ndarray:
    """Apply sigmoid and design-region mask to flattened logits."""
    x = _to_numpy(params, dtype=_np.float64).reshape(self.nely, self.nelx)
    rho = physics.sigmoid(x)
    return (rho * self.mask2d).ravel()

  # ----------------------------- Constraint --------------------------
  def constraint(self, params):
    if self._backend == "pyfantom":
      rho = self._masked_flat(params)
      self._problem.set_desvars(rho.copy())
      gv = self._problem.g()
      # Single-material returns scalar; multi-material returns vector
      return float(gv if _np.isscalar(gv) else gv.reshape(-1)[0])
    # Legacy physics
    x2d = self.reshape(params)
    vol = physics.mean_density(x2d, self.args)
    return vol - self.args["volfrac"]