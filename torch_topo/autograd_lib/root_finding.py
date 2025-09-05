"""
Differentiable root finding algorithms.
"""

from __future__ import annotations

from typing import Callable

import torch


class _FindRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_closure, x, lower, upper, tol, max_iter, *params):
        with torch.no_grad():
            lo = lower.clone()
            hi = upper.clone()
            for _ in range(int(max_iter)):
                y = 0.5 * (lo + hi)
                val = f_closure(x, y, *params)  # scalar/batched scalar
                hi = torch.where(val > 0, y, hi)
                lo = torch.where(val <= 0, y, lo)
                if (hi - lo).abs().max() < tol:
                    break
            y = 0.5 * (lo + hi)
        # Save for backward
        ctx.f = f_closure
        ctx.n_params = len(params)
        ctx.tol = float(tol)
        ctx.save_for_backward(x, y, lower, upper, *params)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        f = ctx.f
        tol = ctx.tol
        x, y, lower, upper, *params = ctx.saved_tensors

        # Rebuild differentiable views
        x_      = x.detach().requires_grad_(x.requires_grad)
        y_      = y.detach().requires_grad_(True)
        lower_  = lower.detach().expand_as(y_)
        upper_  = upper.detach().expand_as(y_)
        params_ = tuple(p.detach().requires_grad_(p.requires_grad) for p in params)

        with torch.enable_grad():
            val = f(x_, y_, *params_)                  # same shape as y_

            # Make sure upstream grad matches shape
            if grad_y.shape != val.shape:
                grad_y = grad_y.reshape(val.shape)

            # --- First pass: df/dy, retain graph if we need a second pass ---
            need_second = x.requires_grad or any(p.requires_grad for p in params_)
            (df_dy,) = torch.autograd.grad(
                val, y_,
                grad_outputs=torch.ones_like(val),
                retain_graph=need_second,   # <-- key change
                allow_unused=False,
            )

            # Sign-preserving, numerically safe denom
            eps = torch.finfo(df_dy.dtype).eps
            denom = torch.where(df_dy.abs() > eps, df_dy, eps * df_dy.sign())

            # Clamp handling: zero gradient at bounds
            interior = ((y_ - lower_) > tol) & ((upper_ - y_) > tol)

            # v = -(∂L/∂y) / (∂f/∂y)
            v = -(grad_y / denom)
            v = torch.where(interior, v, torch.zeros_like(v))

            # --- Second pass: grads w.r.t x and params in ONE call ---
            gx = None
            gparams = ()

            targets = []
            map_idx = []
            if x.requires_grad:
                map_idx.append('x'); targets.append(x_)
            for i, p in enumerate(params_):
                if p.requires_grad:
                    map_idx.append(('p', i)); targets.append(p)

            if targets:
                grads = torch.autograd.grad(
                    val, targets, grad_outputs=v,
                    retain_graph=False, allow_unused=True
                )
                it = iter(grads)
                # unpack in the order we appended to targets
                if x.requires_grad:
                    gx = next(it)
                else:
                    gx = None
                gparams_list = []
                for p in params_:
                    if p.requires_grad:
                        g = next(it)
                        gparams_list.append(torch.zeros_like(p) if g is None else g)
                    else:
                        gparams_list.append(torch.zeros_like(p))
                gparams = tuple(gparams_list)
            else:
                gx = None
                gparams = tuple(torch.zeros_like(p) for p in params_)

        # No grads for f_closure, bounds, tol, max_iter
        return (None, gx, None, None, None, None, *gparams)


def find_root(f_xy: Callable[..., torch.Tensor],
              x: torch.Tensor,
              lower_bound: torch.Tensor,
              upper_bound: torch.Tensor,
              *params: torch.Tensor,
              tolerance: float = 1e-12,
              max_iterations: int = 64) -> torch.Tensor:
    """
    Solve f_xy(x, y, *params) = 0 for scalar y (per batch), assuming monotone in y.
    Differentiable in x and params; returns projection at bounds with zero gradient.
    """
    return _FindRoot.apply(f_xy, x, lower_bound, upper_bound,
                           float(tolerance), int(max_iterations), *params)
