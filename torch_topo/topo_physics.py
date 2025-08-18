# lint as python3
# Torch implementation of topology optimization physics (compliance minimization)

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from contextlib import nullcontext

# Your torch-based autograd_lib (must provide solve_coo, cone_filter, inverse_permutation,
# scatter1d, find_root with torch semantics)
from neural_structural_optimization import autograd_lib

DTYPE = torch.float64

# ---------------------------
# Problem setup / utilities
# ---------------------------

def default_args(device: Optional[torch.device] = None) -> Dict:
    """Match the original defaults; return tensors in torch."""
    nely = 25
    nelx = 80

    # Fix DOFs: left wall (all vertical nodes at x=0) and right-bottom corner (single DOF)
    left_wall = list(range(0, 2 * (nely + 1), 2))
    right_corner = [2 * (nelx + 1) * (nely + 1) - 1]
    fixdofs = torch.as_tensor(left_wall + right_corner, dtype=torch.long, device=device)

    alldofs = torch.arange(2 * (nely + 1) * (nelx + 1), dtype=torch.long, device=device)
    mask_keep = torch.ones_like(alldofs, dtype=torch.bool, device=device)
    mask_keep[fixdofs] = False
    freedofs = torch.tensor(sorted(set(alldofs.tolist()) - set(fixdofs.tolist())),
                        dtype=torch.long, device=device)

    forces = torch.zeros(2 * (nely + 1) * (nelx + 1), dtype=DTYPE, device=device)
    forces[1] = -1.0  # downward unit load at node (0,0) y-DOF

    return {
        'young':       1.0,
        'young_min':   1e-9,
        'poisson':     0.3,
        'g':           0.0,        # gravitational loading off by default
        'volfrac':     0.4,
        'nelx':        nelx,
        'nely':        nely,
        'freedofs':    freedofs,
        'fixdofs':     fixdofs,
        'forces':      forces,
        'mask':        1.0,        # may be scalar or (nely, nelx) tensor
        'penal':       3.0,
        'rmin':        1.5,
        'opt_steps':   50,
        'filter_width': 2,
        'step_size':   0.5,
        'name':        'truss',
        'device':      device if device is not None else forces.device,
        'dtype':       DTYPE,
    }

def _as_mask_tensor(mask, like: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        if mask.shape == like.shape:
            return mask.to(dtype=like.dtype, device=like.device)
        return mask.to(dtype=like.dtype, device=like.device).expand_as(like)
    # scalar -> broadcast
    return torch.as_tensor(mask, dtype=like.dtype, device=like.device).expand_as(like)

# ---------------------------
# Core physics helpers
# ---------------------------

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Stable logistic; matches your tanh-based implementation
    x = torch.clamp(x, -40.0, 40.0)
    return 0.5 * torch.tanh(0.5 * x) + 0.5

def logit(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def sigmoid_with_constrained_mean(x: torch.Tensor, average: float) -> torch.Tensor:
    # Solve for bias b s.t. mean(sigmoid(x + b)) = average, with differentiable root-finding.
    with torch.no_grad():
        lower_bound = logit(torch.as_tensor(average, dtype=x.dtype, device=x.device)) - x.max()
        upper_bound = logit(torch.as_tensor(average, dtype=x.dtype, device=x.device)) - x.min()

    def f(x_, y):
        z = torch.clamp(x_ + y, -40.0, 40.0)
        return sigmoid(z).mean() - average

    b = autograd_lib.find_root(f, x, lower_bound, upper_bound)
    return sigmoid(x + b)

def physical_density(x: torch.Tensor, args: Dict, *, volume_constraint: bool = False, cone_filter: bool = True) -> torch.Tensor:
    # x shape: (nely, nelx)
    x = x.view(args['nely'], args['nelx'])
    if volume_constraint:
        mask = _as_mask_tensor(args['mask'], x) > 0
        x_designed = sigmoid_with_constrained_mean(x[mask], args['volfrac'])
        # scatter back into full field
        x_flat = autograd_lib.scatter1d(x_designed, torch.nonzero(mask.flatten(), as_tuple=False).flatten(), x.numel())
        x = x_flat.view_as(x)
    else:
        x = x * _as_mask_tensor(args['mask'], x)

    if cone_filter:
        x = autograd_lib.cone_filter(x, args['filter_width'], mask=_as_mask_tensor(args['mask'], x))
    return x

def mean_density(x: torch.Tensor, args: Dict, *, volume_constraint: bool = False, cone_filter: bool = True) -> torch.Tensor:
    num = physical_density(x, args, volume_constraint=volume_constraint, cone_filter=cone_filter).mean()
    den = _as_mask_tensor(args['mask'], x).mean()
    return num / den

def get_stiffness_matrix(young: float, poisson: float, *, dtype=DTYPE, device=None) -> torch.Tensor:
    e = torch.as_tensor(young, dtype=dtype, device=device)
    nu = torch.as_tensor(poisson, dtype=dtype, device=device)
    k = torch.tensor([
        1/2 - nu/6,  1/8 + nu/8,  -1/4 - nu/12, -1/8 + 3*nu/8,
       -1/4 + nu/12, -1/8 - nu/8,  nu/6,         1/8 - 3*nu/8
    ], dtype=dtype, device=device)
    ke = torch.stack([
        torch.tensor([k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], dtype=dtype, device=device),
        torch.tensor([k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]], dtype=dtype, device=device),
        torch.tensor([k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]], dtype=dtype, device=device),
        torch.tensor([k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]], dtype=dtype, device=device),
        torch.tensor([k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]], dtype=dtype, device=device),
        torch.tensor([k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]], dtype=dtype, device=device),
        torch.tensor([k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]], dtype=dtype, device=device),
        torch.tensor([k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]], dtype=dtype, device=device),
    ], dim=0)
    return e / (1 - nu**2) * ke

def young_modulus(x: torch.Tensor, e0: float, emin: float, p: float = 3.0) -> torch.Tensor:
    e0_t  = torch.as_tensor(e0, dtype=x.dtype, device=x.device)
    emin_t= torch.as_tensor(emin, dtype=x.dtype, device=x.device)
    return emin_t + (x ** p) * (e0_t - emin_t)

def get_k(stiffness: torch.Tensor, ke: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (value_list, y_list, x_list) for COO assembly."""
    nely, nelx = stiffness.shape
    device = stiffness.device
    dtype  = stiffness.dtype

    # element grid (indexing='ij' => ely: rows [0..nely-1], elx: cols [0..nelx-1])
    ely, elx = torch.meshgrid(
        torch.arange(nely, dtype=torch.long, device=device),
        torch.arange(nelx, dtype=torch.long, device=device),
        indexing='ij'
    )
    # Node numbers per element (shape: (nely, nelx))
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)

    # Element DOFs (per element 8 DOFs; final shape (num_elems, 8))
    edof = torch.stack([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1], dim=0)  # (8, nely, nelx)
    edof = edof.permute(1, 2, 0).reshape(-1, 8)  # (num_elems, 8)

    # COO indices
    x_list = edof.repeat_interleave(8, dim=1).reshape(-1)  # repeat each entry 8x (rows)
    y_list = edof.repeat(1, 8).reshape(-1)                 # tile 8x (cols)

    # Values: (num_elems, 8, 8) times ke
    num_elems = nely * nelx
    kd = stiffness.reshape(num_elems, 1, 1).to(dtype)  # transpose to match element order
    values = (kd * ke.unsqueeze(0)).reshape(-1)            # (num_elems*64,)
    return values, y_list, x_list

# Cheap single-entry cache for index mapping
_dof_cache = {}

def _get_dof_indices(freedofs: torch.Tensor, fixdofs: torch.Tensor,
                     k_xlist: torch.Tensor, k_ylist: torch.Tensor):
    key = (freedofs.data_ptr(), fixdofs.data_ptr(), k_xlist.data_ptr(), k_ylist.data_ptr())
    cached = _dof_cache.get(key)
    if cached is not None:
        return cached

    index_map = autograd_lib.inverse_permutation(torch.cat([freedofs, fixdofs]))  # maps global dof -> position in [free|fixed]
    keep = torch.isin(k_xlist, freedofs) & torch.isin(k_ylist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]
    indices = torch.stack([i, j], dim=0)  # (2, nnz_free)
    _dof_cache[key] = (index_map, keep, indices)
    return _dof_cache[key]

def displace(x_phys: torch.Tensor, ke: torch.Tensor, forces: torch.Tensor,
             freedofs: torch.Tensor, fixdofs: torch.Tensor, *,
             penal: float = 3.0, e_min: float = 1e-9, e_0: float = 1.0) -> torch.Tensor:
    """Solve K u = f under boundary conditions; returns full u with fixed DOFs inserted."""
    stiff = young_modulus(x_phys, e_0, e_min, p=penal)
    k_vals, k_y, k_x = get_k(stiff, ke)

    index_map, keep, indices = _get_dof_indices(freedofs, fixdofs, k_x, k_y)
    # Solve reduced system for free dofs
    u_free = autograd_lib.solve_coo(k_vals[keep], indices, forces[freedofs], sym_pos=True)
    # Rebuild full vector
    u_values = torch.cat([u_free, torch.zeros(fixdofs.numel(), dtype=u_free.dtype, device=u_free.device)])
    return u_values[index_map]

def compliance(x_phys: torch.Tensor, u: torch.Tensor, ke: torch.Tensor, *,
               penal: float = 3.0, e_min: float = 1e-9, e_0: float = 1.0) -> torch.Tensor:
    """Compute compliance = sum_e (E(x)^p * u_e^T ke u_e)."""
    nely, nelx = x_phys.shape
    device = x_phys.device

    ely, elx = torch.meshgrid(
        torch.arange(nely, dtype=torch.long, device=device),
        torch.arange(nelx, dtype=torch.long, device=device),
        indexing='ij'
    )
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)

    all_ixs = torch.stack([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1], dim=0)  # (8, nely, nelx)
    u_sel = u[all_ixs]  # (8, nely, nelx)

    ke_u = torch.einsum('ij,jkl->ikl', ke, u_sel)     # (8, nely, nelx)
    ce   = torch.einsum('ijk,ijk->jk', u_sel, ke_u)   # (nely, nelx)

    E = young_modulus(x_phys, e_0, e_min, p=penal)
    C = (E * ce).sum()
    return C

# ---------------------------
# Optimality criteria 
# ---------------------------
def _smooth_relu(t: torch.Tensor, tau: float) -> torch.Tensor:
    # C¹ smoothing of ReLU using softplus; numerically stable
    # NOTE: assume tau > 0
    return F.softplus(t / tau) * tau

def _soft_clamp(y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float | None):
    if tau is None or tau <= 0:
        return torch.minimum(torch.maximum(y, lo), hi)
    # smooth clamp: lo + softplus(y-lo) - softplus(y-hi)
    return lo + _smooth_relu(y - lo, tau) - _smooth_relu(y - hi, tau)

def optimality_criteria_combine(
    x: torch.Tensor, dc: torch.Tensor, dv: torch.Tensor,
    args: Dict, max_move: float = 0.2, eta: float = 0.5
) -> torch.Tensor:
    volfrac = float(args['volfrac'])
    tau = args.get('smooth_clamp_tau', None)
    if tau is None and torch.is_grad_enabled():
        tau = 1e-3

    # design mask
    m = (_as_mask_tensor(args['mask'], x) > 0)
    m_flat = m.reshape(-1)

    def compute_xnew_from(x0: torch.Tensor, dc0: torch.Tensor, dv0: torch.Tensor, lam: torch.Tensor):
        x0f  = x0.reshape(-1)
        dcf  = dc0.reshape(-1)
        dvf  = dv0.reshape(-1)

        # non-design DOFs: freeze via denominator = 1 (later we also restore x0 there)
        dvf = torch.where(m_flat, dvf, torch.ones_like(dvf))

        # OC multiplicative factor
        ratio_raw = -dcf / (lam * dvf + 1e-30)
        if tau is None or tau <= 0:
            ratio_pos = torch.clamp(ratio_raw, min=0.0)
        else:
            ratio_pos = _smooth_relu(ratio_raw, tau)

        s = ratio_pos.pow(eta)
        xnew = (x0f * s).view_as(x0)

        # move limit, then global [0,1] — both soft when tau>0
        lower = torch.clamp(x0 - max_move, min=0.0)
        upper = torch.clamp(x0 + max_move, max=1.0)
        xnew = _soft_clamp(xnew, lower, upper, tau)
        xnew = _soft_clamp(xnew, x0.new_tensor(0.0), x0.new_tensor(1.0), tau)

        # freeze non-design exactly to x0
        xnew = torch.where(m, xnew, x0)
        return xnew

    # Root eq: mean_density(xnew) == volfrac  (monotone ↑ in lam)
    def f(_dummy: torch.Tensor, lam: torch.Tensor, x0: torch.Tensor, dc0: torch.Tensor, dv0: torch.Tensor):
        xnew = compute_xnew_from(x0, dc0, dv0, lam)
        return volfrac - mean_density(xnew, args)

    # We don’t need the "x" argument of find_root; pass a dummy scalar.
    dummy = x.new_zeros(())
    lam = autograd_lib.find_root(
        f, 
        dummy,
        x.new_tensor(1e-9),
        x.new_tensor(1e9),
        x, dc, dv,
        tolerance=1e-12, max_iterations=64
    )
    return compute_xnew_from(x, dc, dv, lam)


# ---------------------------
# Objective / driver
# ---------------------------

def calculate_forces(x_phys: torch.Tensor, args: Dict) -> torch.Tensor:
    applied = args['forces']
    g = float(args.get('g', 0.0))
    if g == 0.0:
        return applied

    # Average element densities to nodes using 4 shifted pads (exactly as in NumPy version)
    density_node = torch.zeros((x_phys.shape[1] + 1, x_phys.shape[0] + 1), dtype=x_phys.dtype, device=x_phys.device)
    for pad_left in (0, 1):
        for pad_up in (0, 1):
            # pad order for 2D tensors: (left, right, top, bottom)
            padded = F.pad(x_phys.t(), (pad_left, 1 - pad_left, pad_up, 1 - pad_up), value=0.0)
            density_node = density_node + 0.25 * padded

    # Build gravity vector in (nely+1, nelx+1, 2), only y-component nonzero
    density_node = density_node.t()  # to shape (nely+1, nelx+1)
    grav = -g * density_node[..., None] * torch.tensor([0.0, 1.0], dtype=x_phys.dtype, device=x_phys.device)
    return applied + grav.reshape(-1)

def objective(x: torch.Tensor, ke: torch.Tensor, args: Dict, *,
              volume_constraint: bool = False, cone_filter: bool = True) -> torch.Tensor:
    kwargs = dict(penal=args['penal'], e_min=args['young_min'], e_0=args['young'])
    x_phys = physical_density(x, args, volume_constraint=volume_constraint, cone_filter=cone_filter)
    forces = calculate_forces(x_phys, args)
    u = displace(x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs)
    return compliance(x_phys, u, ke, **kwargs)

def optimality_criteria_step(x: torch.Tensor, ke: torch.Tensor, args: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """One OC iteration: returns (compliance, updated_x)."""
    x = x.requires_grad_(True)
    c = objective(x, ke, args)
    (dc,) = torch.autograd.grad(c, x, create_graph=True, retain_graph=True)
    # Use volume_constraint=False for the density gradient to match original implementation
    (dv,) = torch.autograd.grad(mean_density(x, args, volume_constraint=False), x, create_graph=True, retain_graph=True)
    x_new = optimality_criteria_combine(x, dc, dv, args)
    return c, x_new

def run_toposim(x: Optional[torch.Tensor] = None, args: Optional[Dict] = None,
                loss_only: bool = True, verbose: bool = True):
    """Run the full OC loop."""
    if args is None:
        args = default_args()
    device = args.get('device', torch.device('cpu'))

    if x is None:
        x = torch.full((args['nely'], args['nelx']), float(args['volfrac']), dtype=DTYPE, device=device)

    ke = get_stiffness_matrix(args['young'], args['poisson'], dtype=DTYPE, device=device)

    needs_grad = bool(x.requires_grad)
    ctx = nullcontext()

    losses = []
    frames = [x.clone()] if not loss_only else None

    with ctx:
        for step in range(int(args['opt_steps'])):
            c, x = optimality_criteria_step(x, ke, args)
            losses.append(c)
            if not loss_only:
                if verbose and step % 5 == 0:
                    # Safe to log from a detached view
                    print(f"step {step}, loss {c.detach().item():.2e}")
                frames.append(x.clone())

    out = losses[-1] if loss_only else (losses, x, frames)
    return out