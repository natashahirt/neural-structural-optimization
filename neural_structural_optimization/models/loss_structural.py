"""Structural loss function that bridges PyTorch models with NumPy/HIPS-autograd physics."""

import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .utils import batched_topo_loss


class StructuralLoss(torch.autograd.Function):
    """A bridge that lets PyTorch models optimize against NumPy/HIPS-autograd physics."""
    
    @staticmethod
    def forward(ctx, logits: torch.Tensor, env: Any) -> torch.Tensor:
        """Forward pass: convert logits to NumPy and compute physics loss."""
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")

        # Store shape and device/dtype to rebuild grads 
        ctx.input_shape = logits.shape
        ctx.device = logits.device
        ctx.dtype = logits.dtype
        ctx.env = env

        # Save detached tensor for backward 
        logits_cpu = logits.detach().cpu()  # keep original dtype; convert later
        ctx.save_for_backward(logits_cpu)

        # Convert to double NumPy for physics computation
        x_np = logits.detach().cpu().double().numpy()

        # Compute physics losses
        losses_np = batched_topo_loss(x_np, [env])  # -> shape (batch,)

        # Return torch tensor
        return torch.as_tensor(
            np.asarray(losses_np), 
            dtype=torch.float64, 
            device=ctx.device
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass: compute gradients using autograd and map back to PyTorch."""
        (logits_cpu,) = ctx.saved_tensors
        env = ctx.env

        # Convert to NumPy double precision
        x_np = logits_cpu.double().numpy()

        # Compute the gradient of sum(loss_i * grad_output_i) w.r.t. x
        # via HIPS autograd by defining a scalar objective.
        def scalar_objective(x_np: np.ndarray) -> float:
            l = batched_topo_loss(x_np, [env])  # batched loss -> (batch,)
            go = grad_output.detach().cpu().to(torch.float64).numpy()  # apply upstream grad
            return (l * go).sum()

        # Import autograd here to keep top-level file torch-only
        import autograd
        g_np = autograd.grad(scalar_objective)(x_np)  # same shape as x_np

        # Map back to torch, match original dtype & device
        g = torch.from_numpy(g_np).to(ctx.device).to(ctx.dtype)
        return g, None  # no grad for env


# -----------------------------------------------------------------------------
# Torch-native structural loss (no NumPy roundtrip)
# -----------------------------------------------------------------------------

_TORCH_ENV_CACHE: Dict[tuple, Dict[str, torch.Tensor]] = {}


def _inverse_permutation(indices: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(indices)
    out[indices] = torch.arange(indices.numel(), device=indices.device, dtype=indices.dtype)
    return out


def _sigmoid_constrained_mean(x: torch.Tensor, average: float, iters: int = 25) -> torch.Tensor:
    """Shift logits so sigmoid mean equals target volume fraction (diffable Newton)."""
    b = torch.zeros((), device=x.device, dtype=x.dtype)
    avg = x.new_tensor(float(average))
    for _ in range(iters):
        z = torch.clamp(x + b, -40.0, 40.0)
        y = torch.sigmoid(z)
        f = y.mean() - avg
        df = (y * (1 - y)).mean().clamp_min(1e-8)
        b = b - f / df
    return torch.sigmoid(torch.clamp(x + b, -40.0, 40.0))


def _cone_filter(x: torch.Tensor, radius: float, mask: torch.Tensor) -> torch.Tensor:
    """Apply cone filter via normalized convolution."""
    if radius <= 0:
        return x * mask
    r = int(math.ceil(radius))
    device, dtype = x.device, x.dtype
    coords = torch.stack(torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        indexing="ij"
    ), dim=0)
    dist = torch.sqrt(coords[0] ** 2 + coords[1] ** 2)
    kernel = torch.clamp(radius - dist, min=0.0)
    if kernel.sum() <= 0:
        return x * mask
    kernel = kernel / kernel.sum()
    weight = kernel.view(1, 1, 2 * r + 1, 2 * r + 1)

    x_in = x.unsqueeze(0).unsqueeze(0)
    m_in = mask.unsqueeze(0).unsqueeze(0)
    num = F.conv2d(x_in * m_in, weight, padding=r)
    den = F.conv2d(m_in, weight, padding=r).clamp_min(1e-9)
    out = (num / den).squeeze(0).squeeze(0)
    return out * mask


def _assemble_edof(nely: int, nelx: int, device: torch.device) -> torch.Tensor:
    """Element dof indices (num_elems, 8)."""
    ely, elx = torch.meshgrid(
        torch.arange(nely, device=device),
        torch.arange(nelx, device=device),
        indexing="ij",
    )
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    edof = torch.stack([
        2 * n1, 2 * n1 + 1,
        2 * n2, 2 * n2 + 1,
        2 * n3, 2 * n3 + 1,
        2 * n4, 2 * n4 + 1,
    ], dim=0)
    return edof.permute(1, 2, 0).reshape(-1, 8)


def _young_modulus(x: torch.Tensor, e0: float, e_min: float, p: float) -> torch.Tensor:
    return e_min + x.pow(p) * (e0 - e_min)


def _prepare_env(env: Any, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Cache per-env, per-(device,dtype) tensors."""
    key = (id(env), device.type, dtype)
    cached = _TORCH_ENV_CACHE.get(key)
    if cached is not None:
        return cached

    args = env.args
    nelx, nely = int(args["nelx"]), int(args["nely"])
    ndof = 2 * (nely + 1) * (nelx + 1)

    mask_np = args.get("mask", 1)
    if np.isscalar(mask_np):
        mask = torch.ones((nely, nelx), device=device, dtype=dtype) * float(mask_np)
    else:
        mask = torch.as_tensor(mask_np, device=device, dtype=dtype)

    forces = torch.as_tensor(args["forces"], device=device, dtype=dtype)
    freedofs = torch.as_tensor(args["freedofs"], device=device, dtype=torch.long)
    fixdofs = torch.as_tensor(args["fixdofs"], device=device, dtype=torch.long)

    free_flag = torch.zeros(ndof, device=device, dtype=torch.bool)
    free_flag[freedofs] = True

    index_map = _inverse_permutation(torch.cat([freedofs, fixdofs]))

    ke_np = env.ke  # from NumPy
    ke = torch.as_tensor(ke_np, device=device, dtype=dtype)

    edof = _assemble_edof(nely, nelx, device)

    env_dict = {
        "nelx": nelx,
        "nely": nely,
        "mask": mask,
        "forces": forces,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "free_flag": free_flag,
        "index_map": index_map,
        "ke": ke,
        "edof": edof,
        "ndof": ndof,
        "penal": float(args.get("penal", 3.0)),
        "filter_width": float(args.get("filter_width", 0.0)),
        "volfrac": float(args.get("volfrac", 0.5)),
        "young": float(args.get("young", 1.0)),
        "young_min": float(args.get("young_min", 1e-9)),
        "poisson": float(args.get("poisson", 0.3)),
    }
    _TORCH_ENV_CACHE[key] = env_dict
    return env_dict


def _get_stiffness_entries(stiffness: torch.Tensor, ke: torch.Tensor, edof: torch.Tensor):
    """Assemble sparse COO entries (values, row_idx, col_idx)."""
    nelx = stiffness.shape[1]
    kd = stiffness.transpose(0, 1).reshape(-1, 1, 1)
    values = (kd * ke.view(1, 8, 8)).reshape(-1)

    # Build global index lists
    x_list = edof.repeat_interleave(8, dim=1).reshape(-1)
    y_list = edof.repeat(1, 8).reshape(-1)
    return values, y_list, x_list


def _cg_solve(mat_values: torch.Tensor,
              mat_indices: torch.Tensor,
              size: int,
              b: torch.Tensor,
              tol: float = 1e-7,
              max_iter: int = 400,
              precondition: bool = True) -> torch.Tensor:
    """Preconditioned CG (Jacobi) for SPD sparse matrix given COO entries."""
    A = torch.sparse_coo_tensor(mat_indices, mat_values, (size, size))
    x = torch.zeros_like(b)
    r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)

    if precondition:
        # Jacobi preconditioner: M^{-1} = 1 / diag(A)
        diag = torch.zeros(size, device=b.device, dtype=b.dtype)
        i, j = mat_indices
        diag_mask = (i == j)
        if diag_mask.any():
            diag.scatter_add_(0, i[diag_mask], mat_values[diag_mask])
        diag = diag.clamp_min(1e-12)
        Minv = 1.0 / diag
        z = Minv * r
        p = z.clone()
        rsold = torch.dot(r, z)
    else:
        p = r.clone()
        rsold = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
        denom = torch.dot(p, Ap).clamp_min(1e-12)
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if precondition:
            z = Minv * r
            rsnew = torch.dot(r, z)
        else:
            rsnew = torch.dot(r, r)

        if torch.sqrt(rsnew) < tol:
            break

        beta = rsnew / rsold
        p = (z if precondition else r) + beta * p
        rsold = rsnew
    return x


def _compliance(x_phys: torch.Tensor,
                u: torch.Tensor,
                ke: torch.Tensor,
                nely: int,
                nelx: int,
                e0: float,
                e_min: float,
                penal: float) -> torch.Tensor:
    """Compliance using vectorized einsums."""
    device = x_phys.device
    ely, elx = torch.meshgrid(
        torch.arange(nely, device=device),
        torch.arange(nelx, device=device),
        indexing="ij",
    )
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    all_ixs = torch.stack([
        2 * n1, 2 * n1 + 1,
        2 * n2, 2 * n2 + 1,
        2 * n3, 2 * n3 + 1,
        2 * n4, 2 * n4 + 1,
    ], dim=0).reshape(8, -1)

    u_sel = u[all_ixs]  # (8, num_elem)
    ke_u = torch.matmul(ke, u_sel)
    ce = (u_sel * ke_u).sum(dim=0)
    C = _young_modulus(x_phys, e0, e_min, penal).reshape(-1) * ce
    return C.sum()


def torch_structural_loss(
    logits: torch.Tensor,
    env: Any,
    *,
    volume_constraint: bool = True,
    cone_filter: bool = True,
    cg_tol: float = 1e-7,
    cg_max_iter: int = 400,
) -> torch.Tensor:
    """Torch-native structural loss; returns per-sample loss tensor."""
    if logits.dim() == 4 and logits.size(1) == 1:
        logits = logits[:, 0]
    if logits.dim() == 3:
        batch, nely, nelx = logits.shape
    else:
        raise ValueError("logits must have shape (B, H, W) or (B,1,H,W)")

    device = logits.device
    # Work in float64 for stability; convert env tensors accordingly.
    logits64 = logits.to(torch.float64)
    tenv = _prepare_env(env, device, torch.float64)
    if (tenv["nely"], tenv["nelx"]) != (nely, nelx):
        raise ValueError(f"logit grid {nely}x{nelx} does not match env {tenv['nely']}x{tenv['nelx']}")

    losses = []
    for b in range(batch):
        x = logits64[b]
        mask = tenv["mask"]

        if volume_constraint:
            design = torch.zeros_like(x)
            active = mask > 0
            if active.any():
                design_active = _sigmoid_constrained_mean(x[active], tenv["volfrac"])
                design = design_active.new_zeros(x.shape)
                design[active] = design_active
        else:
            design = torch.sigmoid(x) * mask

        if cone_filter:
            design = _cone_filter(design, tenv["filter_width"], mask)

        stiffness = _young_modulus(design, tenv["young"], tenv["young_min"], tenv["penal"])
        k_entries, k_ylist, k_xlist = _get_stiffness_entries(stiffness, tenv["ke"], tenv["edof"])

        free_flag = tenv["free_flag"]
        keep = free_flag[k_xlist] & free_flag[k_ylist]
        indices = torch.stack([
            tenv["index_map"][k_ylist[keep]],
            tenv["index_map"][k_xlist[keep]],
        ], dim=0)
        values = k_entries[keep]

        bvec = tenv["forces"][tenv["freedofs"]]
        u_free = _cg_solve(
            values,
            indices,
            tenv["freedofs"].numel(),
            bvec,
            tol=cg_tol,
            max_iter=cg_max_iter,
            precondition=True,
        )

        u_full = torch.zeros(tenv["ndof"], device=device, dtype=torch.float64)
        u_full[tenv["freedofs"]] = u_free

        c = _compliance(
            design,
            u_full,
            tenv["ke"],
            tenv["nely"],
            tenv["nelx"],
            tenv["young"],
            tenv["young_min"],
            tenv["penal"],
        )
        losses.append(c)

    return torch.stack(losses, dim=0)
