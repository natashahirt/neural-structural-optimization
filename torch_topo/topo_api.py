# lint as python3
# PyTorch-native wrapper around physics

from __future__ import annotations
from typing import Any, Dict, Optional, Union

try:
    import torch
except Exception as e:
    raise RuntimeError("This module requires PyTorch.") from e

import numpy as _np  # only for tolerant boundary input ingestion
from neural_structural_optimization import physics

DTYPE = torch.float64  # keep double for stability, as in your pipeline


# ---------------------------
# Helpers
# ---------------------------

def _is_scalar_like(x: Any) -> bool:
    """Check if x is scalar-like (0-d tensor, numpy scalar, or Python numeric)."""
    if torch.is_tensor(x):
        return x.ndim == 0
    if isinstance(x, (_np.generic, int, float, bool)):
        return True
    return False


def _to_torch(x: Any, *, dtype: torch.dtype = DTYPE,
              device: Optional[torch.device] = None,
              detach_constants: bool = True) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
    elif hasattr(x, "values"):
        t = torch.as_tensor(_np.asarray(x.values))
    elif isinstance(x, (list, tuple)) and len(x) == 0:
        raise ValueError("Cannot convert empty sequence to tensor")
    else:
        try:
            t = torch.as_tensor(x)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot convert {type(x)} to tensor: {e}") from e

    if detach_constants and t.requires_grad:
        t = t.detach()
    if device is not None:
        t = t.to(device)

    # Dtype conversion (only for floating point)
    if dtype is not None and t.dtype.is_floating_point and t.dtype != dtype:
        t = t.to(dtype)

    return t

def _args_to_torch(
    args: Dict[str, Any],
    *,
    dtype: torch.dtype = DTYPE,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Convert args to torch tensors, keep scalars as Python values."""
    out: Dict[str, Any] = dict(args)

    # Array-like fields
    for key in ("mask", "freedofs", "fixdofs", "forces"):
        if key in out:
            if key in ("freedofs", "fixdofs"):
                out[key] = _to_torch(
                    out[key], 
                    dtype=torch.long, 
                    device=device, 
                    detach_constants=True
                ).view(-1)
            elif key == "forces":
                out[key] = _to_torch(
                    out[key], 
                    dtype=dtype, 
                    device=device, 
                    detach_constants=True
                ).view(-1)
            else:  # mask
                out[key] = _to_torch(
                    out[key], 
                    dtype=dtype, 
                    device=device, 
                    detach_constants=True
                )

    # Scalar-ish fields: normalize to plain Python scalars
    scalar_keys = (
        "young", "young_min", "poisson", "g",
        "volfrac", "xmin", "xmax", "nelx", "nely",
        "penal", "filter_width"
    )
    for key in scalar_keys:
        if key in out:
            v = out[key]
            if torch.is_tensor(v):
                if v.numel() != 1:
                    raise ValueError(f"Expected scalar for {key}, got tensor of shape {v.shape}")
                v = v.detach().cpu().item()
            elif hasattr(v, "item") and not isinstance(v, (bytes, str)):
                try:
                    v = v.item()
                except Exception:
                    pass
            out[key] = v

    return out


# ---------------------------
# Torch-spec problem setup
# ---------------------------

def specified_task(
    problem: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = DTYPE
) -> Dict[str, Any]:
    """Convert problem to torch-compatible parameters."""
    nelx = int(problem.width)
    nely = int(problem.height)
    
    if nelx <= 0 or nely <= 0:
        raise ValueError(f"Invalid dimensions: nelx={nelx}, nely={nely}")

    # Convert inputs to torch tensors
    normals = _to_torch(
        problem.normals, 
        dtype=torch.float64, 
        device=device, 
        detach_constants=True
    ).view(-1)
    
    forces = _to_torch(
        problem.forces, 
        dtype=dtype, 
        device=device, 
        detach_constants=True
    ).view(-1)

    # Handle mask (scalar or tensor)
    if _is_scalar_like(problem.mask):
        mask = float(problem.mask)
    else:
        mask = _to_torch(
            problem.mask, 
            dtype=dtype, 
            device=device, 
            detach_constants=True
        )

    # Fixed DOFs where normals != 0 (match original semantics)
    fixed_bool = normals != 0
    fixdofs = torch.nonzero(fixed_bool, as_tuple=False).view(-1).to(torch.long)

    # Free DOFs as complement
    ndof = 2 * (nelx + 1) * (nely + 1)
    alldofs = torch.arange(ndof, device=device, dtype=torch.long)
    freedofs = alldofs[~fixed_bool]  # complement

    # Validate force vector length
    if forces.numel() != ndof:
        raise ValueError(
            f"Force vector length {forces.numel()} != expected {ndof}"
        )

    # Validate normals vector length
    if normals.numel() != ndof:
        raise ValueError(
            f"Normals vector length {normals.numel()} != expected {ndof}"
        )

    params: Dict[str, Any] = {
        # Material properties
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0.0,
        # Constraints
        "volfrac": float(problem.density),
        "xmin": 0.001,
        "xmax": 1.0,
        # Input parameters
        "nelx": nelx,
        "nely": nely,
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": forces,
        "penal": 3.0,
        "filter_width": 2,
    }
    return params


# ---------------------------
# Torch Environment wrapper
# ---------------------------

class Environment:
    """Torch wrapper around physics engine."""

    def __init__(
        self,
        args: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DTYPE,
        persist_ke: bool = True
    ):
        """Initialize with problem parameters."""
        self.device = device if device is not None else args.get("device", None)
        self.dtype = dtype

        # Normalize args to torch once
        self.args = _args_to_torch(args, dtype=self.dtype, device=self.device)

        # Stiffness matrix (torch). Handle either torch or numpy return from physics.
        ke = physics.get_stiffness_matrix(self.args["young"], self.args["poisson"])
        if not torch.is_tensor(ke):
            ke = torch.as_tensor(ke, dtype=self.dtype, device=self.device)
        else:
            ke = ke.to(dtype=self.dtype, device=self.device)
        self.ke = ke if persist_ke else None  # optional caching

    @property
    def nelx(self) -> int:
        """Elements in x direction."""
        return int(self.args["nelx"])

    @property
    def nely(self) -> int:
        """Elements in y direction."""
        return int(self.args["nely"])

    @property
    def ndof(self) -> int:
        """Total degrees of freedom."""
        return 2 * (self.nelx + 1) * (self.nely + 1)

    def reshape(self, params: Union[torch.Tensor, _np.ndarray]) -> torch.Tensor:
        """Reshape to (nely, nelx), preserving gradients."""
        if not torch.is_tensor(params):
            x = torch.as_tensor(params, dtype=self.dtype, device=self.device)
        else:
            x = params.to(self.dtype)
            if self.device is not None:
                x = x.to(self.device)
        
        # Handle reshaping
        if x.ndim == 1:
            expected_size = self.nely * self.nelx
            if x.numel() != expected_size:
                raise ValueError(
                    f"1D params size {x.numel()} != expected {expected_size}"
                )
            x = x.view(self.nely, self.nelx)
        elif x.shape != (self.nely, self.nelx):
            try:
                x = x.reshape(self.nely, self.nelx)
            except RuntimeError as e:
                raise ValueError(
                    f"Cannot reshape params of shape {x.shape} to ({self.nely}, {self.nelx})"
                ) from e
        
        return x

    def render(
        self, 
        params: Union[torch.Tensor, _np.ndarray], 
        volume_constraint: bool = True
    ) -> torch.Tensor:
        """Apply physical density mapping."""
        x2d = self.reshape(params)
        return physics.physical_density(
            x2d,
            self.args,
            volume_constraint=volume_constraint,
            cone_filter=False,  # mirror NumPy version
        )

    def objective(
        self, 
        params: Union[torch.Tensor, _np.ndarray], 
        volume_constraint: bool = False
    ) -> torch.Tensor:
        """Compute compliance objective."""
        x2d = self.reshape(params)
        
        # Get stiffness matrix (cached or compute)
        ke = self.ke
        if ke is None:  # if not persisted
            ke = physics.get_stiffness_matrix(self.args["young"], self.args["poisson"])
            if not torch.is_tensor(ke):
                ke = torch.as_tensor(ke, dtype=self.dtype, device=self.device)
            else:
                ke = ke.to(dtype=self.dtype, device=self.device)

        return physics.objective(
            x2d,
            ke,
            self.args,
            volume_constraint=volume_constraint,
            cone_filter=True,  # mirror NumPy version
        )

    def constraint(self, params: Union[torch.Tensor, _np.ndarray]) -> torch.Tensor:
        """Compute volume constraint violation."""
        x2d = self.reshape(params)
        vol = physics.mean_density(x2d, self.args)
        # Return a scalar tensor matching autograd expectations
        return vol - float(self.args["volfrac"])

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Environment(nelx={self.nelx}, nely={self.nely}, "
            f"device={self.device}, dtype={self.dtype})"
        )

    def validate_setup(self) -> None:
        """Validate problem setup consistency."""
        # Check dimensions
        if self.nelx <= 0 or self.nely <= 0:
            raise ValueError(f"Invalid dimensions: nelx={self.nelx}, nely={self.nely}")
        
        # Check force vector length
        expected_force_len = 2 * (self.nelx + 1) * (self.nely + 1)
        actual_force_len = self.args["forces"].numel()
        if actual_force_len != expected_force_len:
            raise ValueError(
                f"Force vector length {actual_force_len} != expected {expected_force_len}"
            )
        
        # Check DOF consistency
        freedofs = self.args["freedofs"]
        fixdofs = self.args["fixdofs"]
        
        if freedofs.numel() + fixdofs.numel() != self.ndof:
            raise ValueError(
                f"DOF mismatch: {freedofs.numel()} + {fixdofs.numel()} != {self.ndof}"
            )
        
        # Check for overlapping DOFs
        if torch.any(torch.isin(freedofs, fixdofs)):
            raise ValueError("Found overlapping free and fixed DOFs")
        
        # Check volume fraction
        volfrac = self.args["volfrac"]
        if not (0.0 < volfrac <= 1.0):
            raise ValueError(f"Invalid volume fraction: {volfrac}")

    def get_problem_info(self) -> Dict[str, Any]:
        """Get problem setup summary."""
        return {
            "nelx": self.nelx,
            "nely": self.nely,
            "ndof": self.ndof,
            "n_free_dofs": self.args["freedofs"].numel(),
            "n_fixed_dofs": self.args["fixdofs"].numel(),
            "volfrac": self.args["volfrac"],
            "device": self.device,
            "dtype": self.dtype,
        }

    def create_initial_design(self, *, method: str = "uniform") -> torch.Tensor:
        """Create initial design vector."""
        size = self.nely * self.nelx
        
        if method == "uniform":
            # Uniform density at volume fraction
            return torch.full(
                (size,), 
                self.args["volfrac"], 
                dtype=self.dtype, 
                device=self.device
            )
        elif method == "random":
            # Random density in [0, 1]
            return torch.rand(size, dtype=self.dtype, device=self.device)
        elif method == "ones":
            return torch.ones(size, dtype=self.dtype, device=self.device)
        elif method == "zeros":
            return torch.zeros(size, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def compute_gradients(
        self, 
        params: torch.Tensor, 
        *, 
        create_graph: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute objective and constraint gradients."""
        if not params.requires_grad:
            raise ValueError("params must have requires_grad=True")

        obj = self.objective(params)
        if not torch.isfinite(obj):
            raise FloatingPointError(f"Objective is not finite: {obj.item()}")
        obj.backward(retain_graph=True, create_graph=create_graph)
        obj_grad = params.grad.clone()
        params.grad.zero_()

        con = self.constraint(params)
        if not torch.isfinite(con):
            raise FloatingPointError(f"Constraint is not finite: {con.item()}")
        con.backward(retain_graph=False, create_graph=create_graph)
        con_grad = params.grad.clone()
        params.grad.zero_()

        return {"objective_grad": obj_grad, "constraint_grad": con_grad,
                "objective": obj.detach(), "constraint": con.detach()}
