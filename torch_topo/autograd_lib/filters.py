"""
Filtering operations for topology optimization.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

from .utils import Tensor, Device


# ==========================================
# Gaussian
# ==========================================

@lru_cache(None)
def _gaussian_kernel_1d(sigma: float, dtype: torch.dtype, device: Device) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    k = k / k.sum()
    return k


def _circ_pad_lr(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Circular pad along the last (W) dimension for any pad size."""
    if left == 0 and right == 0: 
        return x
    W = x.size(-1)
    idx = (torch.arange(-left, W + right, device=x.device) % W)
    return x.index_select(-1, idx)


def _circ_pad_tb(x: torch.Tensor, top: int, bottom: int) -> torch.Tensor:
    """Circular pad along the second-to-last (H) dimension for any pad size."""
    if top == 0 and bottom == 0:
        return x
    H = x.size(-2)
    idx = (torch.arange(-top, H + bottom, device=x.device) % H)
    return x.index_select(-2, idx)


class _GaussianFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, width: float):
        # Normalize to (N,1,H,W)
        if x.dim() == 2:
            x4 = x.unsqueeze(0).unsqueeze(0)
            squeeze_2d = True
        elif x.dim() == 3:
            x4 = x.unsqueeze(1)
            squeeze_2d = False
        else:
            raise AssertionError(f"Expected 2d or 3d input, got {x.shape}")

        if width <= 0:
            y = x4
            k = None
            pad = 0
        else:
            dtype, device = x4.dtype, x4.device
            k = _gaussian_kernel_1d(float(width), dtype, device)
            pad = (k.numel() - 1) // 2
            # horizontal then vertical, both circular
            y = F.conv2d(_circ_pad_lr(x4, pad, pad), k.view(1, 1, 1, -1))
            y = F.conv2d(_circ_pad_tb(y,  pad, pad), k.view(1, 1, -1, 1))

        ctx.width = float(width)
        ctx.squeeze_2d = squeeze_2d
        # Nothing else needed; backward will recompute k with same width.
        return y[0, 0] if squeeze_2d else y[:, 0]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # VJP for symmetric kernel + circular BCs is the SAME filter.
        if ctx.width <= 0:
            return grad_out.clone(), None

        # Normalize grad to (N,1,H,W)
        if grad_out.dim() == 2:
            g4 = grad_out.unsqueeze(0).unsqueeze(0)
            squeeze_2d = True
        elif grad_out.dim() == 3:
            g4 = grad_out.unsqueeze(1)
            squeeze_2d = False
        else:
            raise AssertionError(f"Expected 2d or 3d grad, got {grad_out.shape}")

        dtype, device = g4.dtype, g4.device
        k = _gaussian_kernel_1d(ctx.width, dtype, device)
        pad = (k.numel() - 1) // 2

        gy = F.conv2d(_circ_pad_lr(g4, pad, pad), k.view(1, 1, 1, -1))
        gy = F.conv2d(_circ_pad_tb(gy,  pad, pad), k.view(1, 1, -1, 1))

        gx = gy[0, 0] if squeeze_2d else gy[:, 0]
        return gx, None

def gaussian_filter(x: torch.Tensor, width: float) -> torch.Tensor:
    """Separable 2D Gaussian with circular boundaries (sum-preserving kernel)."""
    return _GaussianFilter.apply(x, float(width))


# ==========================================
# Cone filter
# ==========================================

def _f_flatten(x2d: torch.Tensor) -> torch.Tensor:
    # Fortran-style flatten: (nely, nelx) -> (nelx*nely,)
    return x2d.t().reshape(-1)


def _f_unflatten(v: torch.Tensor, nely: int, nelx: int) -> torch.Tensor:
    # Inverse of _f_flatten: (nelx*nely,) -> (nely, nelx)
    return v.view(nelx, nely).t()


def _cone_filter_matrix_from_mask(
    nelx: int, nely: int, radius: float, m_bool: torch.Tensor,
    device: torch.device, dtype: torch.dtype
):
    """
    Build column-normalized cone filter H_col on `device`/`dtype`,
    using a boolean mask tensor `m_bool` of shape (nely, nelx) ON THE SAME DEVICE.
    Returns (indices [2,nnz] long, values [nnz] dtype, shape=(n,n)).
    """
    assert m_bool.shape == (nely, nelx)
    m_bool = m_bool.to(device=device, dtype=torch.bool)

    n = nelx * nely
    # Grid (device)
    xg = torch.arange(nelx, device=device, dtype=torch.long)
    yg = torch.arange(nely, device=device, dtype=torch.long)
    X, Y = torch.meshgrid(xg, yg, indexing='ij')  # (nelx, nely)

    r_bound = int(math.ceil(float(radius)))
    rows, cols, vals = [], [], []

    for dx in range(-r_bound, r_bound + 1):
        for dy in range(-r_bound, r_bound + 1):
            w = float(radius) - math.hypot(dx, dy)
            if w <= 0.0:
                continue
            Xj = X + dx
            Yj = Y + dy
            valid = (
                m_bool.t() &
                (Xj >= 0) & (Xj < nelx) &
                (Yj >= 0) & (Yj < nely)
            )
            if not bool(valid.any()):
                continue

            # Flatten (Fortran): col-major = Y + nely * X
            row = (Y + nely * X)[valid]           # output center index
            col = (Yj + nely * Xj)[valid]         # input neighbor index

            rows.append(row)
            cols.append(col)
            vals.append(torch.full(row.shape, w, dtype=dtype, device=device))

    if not rows:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        val = torch.empty((0,),    dtype=dtype,       device=device)
        return idx, val, (n, n)

    rows = torch.cat(rows).long()
    cols = torch.cat(cols).long()
    vals = torch.cat(vals).to(dtype)

    # Column-normalize without sparse.to_dense()
    colsum = torch.zeros(n, dtype=dtype, device=device)
    colsum.index_add_(0, cols, vals)
    invcol = torch.zeros_like(colsum)
    nz = colsum != 0
    invcol[nz] = 1.0 / colsum[nz]
    vals = vals * invcol.index_select(0, cols)

    idx = torch.stack([rows, cols], dim=0)  # (2, nnz)
    return idx, vals, (n, n)


# Cache ====================================

_CONE_DEV_CACHE_MAX = 64
_CONE_DEV_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, tuple[int,int]]]" = OrderedDict()

def _mask_hash64(m_bool: torch.Tensor) -> int:
    """Cheap 64-bit hash computed on device; only the final scalar crosses to CPU."""
    u8 = m_bool.to(torch.uint8).flatten().to(torch.uint64)
    # 64-bit mix (golden ratio multiplier); modulo 2^64 via uint64 dtype
    h = (u8 * 0x9E3779B97F4A7C15).sum(dtype=torch.uint64)
    return int(h.item())  # single 8-byte host read

def _cone_cache_get_or_build(
    nelx: int, nely: int, radius: float, m_bool: torch.Tensor,
    device: torch.device, dtype: torch.dtype
):
    # Optional special keys for all-true/all-false masks to reduce hashing overhead
    mt = m_bool
    all_true  = bool(mt.all().item())
    all_false = not all_true and bool((~mt).all().item())
    mkey = ("ALL1" if all_true else ("ALL0" if all_false else _mask_hash64(mt)))

    key = (nelx, nely, float(radius), mkey, device.type, device.index, str(dtype))
    hit = _CONE_DEV_CACHE.get(key)
    if hit is not None:
        _CONE_DEV_CACHE.move_to_end(key)
        return hit

    # Build on device and insert
    idx_base, val, shape = _cone_filter_matrix_from_mask(nelx, nely, radius, m_bool, device, dtype)
    _CONE_DEV_CACHE[key] = (idx_base, val, shape)
    if len(_CONE_DEV_CACHE) > _CONE_DEV_CACHE_MAX:
        _CONE_DEV_CACHE.popitem(last=False)  # evict LRU
    return idx_base, val, shape

def clear_cone_cache():
    _CONE_DEV_CACHE.clear()


# Class ====================================

class _ConeFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                inputs: torch.Tensor,
                radius: float,
                mask: Optional[torch.Tensor] = 1,
                transpose: bool = False) -> torch.Tensor:
        if inputs.dim() != 2:
            raise ValueError(f"cone_filter expects (nely, nelx), got {tuple(inputs.shape)}")

        nely, nelx = int(inputs.shape[0]), int(inputs.shape[1])
        device, dtype = inputs.device, inputs.dtype

        # Prepare mask ON DEVICE
        if isinstance(mask, torch.Tensor):
            m = mask.to(device=device, dtype=torch.bool)
            if m.shape != (nely, nelx):
                m = m.expand(nely, nelx)
        else:
            m = torch.full((nely, nelx), bool(mask), dtype=torch.bool, device=device)

        # >>> cached device/dtype-aware build <<<
        idx_base, val, shape = _cone_cache_get_or_build(nelx, nely, float(radius), m, device, dtype)

        # Use F or F^T in forward
        idx_use = idx_base if not transpose else idx_base[[1, 0]]
        F = torch.sparse_coo_tensor(idx_use, val, shape, dtype=dtype, device=device).coalesce()

        # y = (F x)_F-order, then gate outputs by mask
        x_vec = _f_flatten(inputs.to(dtype))
        y_vec = torch.sparse.mm(F, x_vec[:, None]).squeeze(1)
        y = _f_unflatten(y_vec, nely, nelx)
        y = y * m.to(dtype)

        # Save base operator & mask (device tensors)
        ctx.save_for_backward(idx_base, val, m)
        ctx.shape = shape
        ctx.transpose_used = bool(transpose)
        ctx.dims = (nely, nelx)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        idx_base, val, m = ctx.saved_tensors
        nely, nelx = ctx.dims
        device, dtype = grad_out.device, grad_out.dtype

        # Mask the upstream gradient (output gating)
        g = grad_out * m.to(dtype)

        # Flatten in Fortran order
        g_vec = _f_flatten(g)

        # Adjoint operator: swap if forward used base or transpose
        idx_adj = idx_base[[1, 0]] if not ctx.transpose_used else idx_base
        F_adj = torch.sparse_coo_tensor(idx_adj, val, ctx.shape, dtype=dtype, device=device).coalesce()

        # grad_x = F_adj * g
        gx_vec = torch.sparse.mm(F_adj, g_vec[:, None]).squeeze(1)
        gx = _f_unflatten(gx_vec, nely, nelx)

        # No grads for radius/mask/transpose
        return gx, None, None, None


def cone_filter(inputs: torch.Tensor, radius: float,
                mask: Optional[torch.Tensor] = 1,
                transpose: bool = False) -> torch.Tensor:
    """
    Cone filter (column-normalized, zero-boundary, Fortran flatten) with custom VJP.
    Entire build and apply occur on the input's device/dtype.
    """
    return _ConeFilter.apply(inputs, float(radius), mask, bool(transpose))