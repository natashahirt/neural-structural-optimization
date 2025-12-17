"""
Sparse matrix solvers for structural analysis.
"""

from __future__ import annotations

import torch

from .utils import Tensor, normalize_sparse_indices


def _spmv_coo(rows: Tensor, cols: Tensor, vals: Tensor, v: Tensor, n: int) -> Tensor:
    Av = vals[:, None] * v.index_select(0, cols)   # (nnz,k)
    y = v.new_zeros(n, v.shape[1])
    y.index_add_(0, rows, Av)
    return y


def _spmv_coo_T(rows: Tensor, cols: Tensor, vals: Tensor, v: Tensor, n: int) -> Tensor:
    """Transpose sparse matrix-vector product: A^T * v"""
    Av = vals[:, None] * v.index_select(0, rows)   # (nnz,k) - note: rows instead of cols
    y = v.new_zeros(n, v.shape[1])
    y.index_add_(0, cols, Av)                      # note: cols instead of rows
    return y


def _jacobi_diag(rows: Tensor, cols: Tensor, vals: Tensor, n: int) -> Tensor:
    d = vals.new_zeros(n)
    mask = rows.eq(cols)
    if mask.any():
        d.index_add_(0, rows[mask], vals[mask])
    return d


def _pcg(Amv, b: Tensor, M_inv: Tensor | None, tol=1e-8, maxiter=2000):
    n, k = b.shape
    x = b.new_zeros(n, k)
    r = b - Amv(x)
    if M_inv is None:
        z = r
        rs = (r * r).sum(0, keepdim=True)
    else:
        z = M_inv[:, None] * r
        rs = (r * z).sum(0, keepdim=True)
    p = z.clone()
    bnorm = torch.linalg.vector_norm(b, dim=0, keepdim=True).clamp_min(1e-30)
    for _ in range(maxiter):
        Ap = Amv(p)
        denom = (p * Ap).sum(0, keepdim=True).clamp_min(1e-30)
        alpha = rs / denom
        x = x + p * alpha
        r = r - Ap * alpha
        rel = torch.linalg.vector_norm(r, dim=0, keepdim=True) / bnorm
        if bool((rel < tol).all()):
            break
        if M_inv is None:
            z = r
            rs_new = (r * r).sum(0, keepdim=True)
        else:
            z = M_inv[:, None] * r
            rs_new = (r * z).sum(0, keepdim=True)
        beta = rs_new / rs
        p = z + p * beta
        rs = rs_new
    return x


def _bicgstab(Amv, b: Tensor, M_inv: Tensor | None, tol=1e-8, maxiter=2000):
    n, k = b.shape
    x = b.new_zeros(n, k)
    r = b - Amv(x)
    r_hat = r.clone()
    v = b.new_zeros(n, k)
    p = b.new_zeros(n, k)
    rho = torch.ones(1, k, dtype=b.dtype, device=b.device)
    alpha = torch.ones(1, k, dtype=b.dtype, device=b.device)
    omega = torch.ones(1, k, dtype=b.dtype, device=b.device)
    bnorm = torch.linalg.vector_norm(b, dim=0, keepdim=True).clamp_min(1e-30)

    for _ in range(maxiter):
        rho_new = (r_hat * r).sum(0, keepdim=True).clamp_min(1e-30)
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)

        y = p if M_inv is None else M_inv[:, None] * p
        v = Amv(y)
        den = (r_hat * v).sum(0, keepdim=True).clamp_min(1e-30)
        alpha = rho_new / den

        s = r - alpha * v
        x_tmp = x + alpha * y
        rel_s = torch.linalg.vector_norm(s, dim=0, keepdim=True) / bnorm
        if bool((rel_s < tol).all()):
            x = x_tmp
            break

        z = s if M_inv is None else M_inv[:, None] * s
        t = Amv(z)
        tt = (t * t).sum(0, keepdim=True).clamp_min(1e-30)
        omega = ((t * s).sum(0, keepdim=True) / tt)
        x = x_tmp + omega * z
        r = s - omega * t

        rel = torch.linalg.vector_norm(r, dim=0, keepdim=True) / bnorm
        if bool((rel < tol).all()):
            break
        rho = rho_new
    return x


class _SolveCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                a_indices: Tensor, a_values: Tensor, size: int, b: Tensor,
                sym_pos: bool, tol: float, maxiter: int,
                use_jacobi: bool, dense_threshold: int, dense_density: float):
        single = (b.dim() == 1)
        if single:
            b = b[:, None]

        n = int(size)
        if a_indices.dtype != torch.long:
            a_indices = a_indices.long()

        # keep original order
        a_indices = normalize_sparse_indices(a_indices)
        rows0 = a_indices[0].to(b.device).contiguous()
        cols0 = a_indices[1].to(b.device).contiguous()
        vals0 = a_values.to(b.dtype).to(b.device).contiguous()

        # build & coalesce once
        A_coo = torch.sparse_coo_tensor(
            torch.stack((rows0, cols0), 0), vals0, (n, n),
            dtype=b.dtype, device=b.device
        ).coalesce()

        idx = A_coo.indices()
        rows, cols, vals = idx[0], idx[1], A_coo.values()
        nnz = vals.numel()
        density = nnz / float(n * n)

        # --- map original -> coalesced order ---
        # flatten keys (row-major)
        keys0 = rows0 * n + cols0            # (nnz_orig,)
        keys  = rows  * n + cols             # (nnz_coal,)
        sorted_keys, perm = torch.sort(keys) # perm: positions in 'keys'
        pos = torch.searchsorted(sorted_keys, keys0)
        idx_map = perm[pos]                  # (nnz_orig,) each -> index in coalesced

        # Preconditioner (Jacobi)
        M_inv = None
        if use_jacobi:
            # Diagonal from coalesced A
            diag = torch.zeros(n, dtype=b.dtype, device=b.device)
            same = (rows == cols)
            if same.any():
                diag.index_add_(0, rows[same], vals[same])
            M_inv = diag.clamp_min(1e-30).reciprocal()

        # Dense shortcut
        use_dense = (n <= dense_threshold) or (density >= dense_density)
        if use_dense:
            A = A_coo.to_dense()
            u = torch.linalg.solve(A, b)
            ctx.save_for_backward(rows, cols, vals, u, A, idx_map)
            use_transpose_bwd = (not sym_pos)
            ctx.flags = (n, use_transpose_bwd, tol, maxiter, use_jacobi, True, single)
        else:
            Amv = lambda x: _spmv_coo(rows, cols, vals, x, n)
            if sym_pos:
                u = _pcg(Amv, b, M_inv, tol=tol, maxiter=maxiter)
                use_transpose_bwd = False
            else:
                u = _bicgstab(Amv, b, M_inv, tol=tol, maxiter=maxiter)
                use_transpose_bwd = True
            M_inv_saved = M_inv if (M_inv is not None) else torch.tensor([], device=b.device, dtype=b.dtype)
            ctx.save_for_backward(rows, cols, vals, u, M_inv_saved, idx_map)
            ctx.flags = (n, use_transpose_bwd, tol, maxiter, use_jacobi, False, single)

        return u.squeeze(1) if single else u

    @staticmethod
    def backward(ctx, grad_u: Tensor):
        rows, cols, vals, u, extra, idx_map = ctx.saved_tensors
        n, use_transpose_bwd, tol, maxiter, use_jacobi, use_dense, single_rhs = ctx.flags

        if grad_u.dim() == 1:
            grad_u = grad_u[:, None]
        if u.dim() == 1:
            u = u[:, None]

        if use_dense:
            A = extra
            lam = torch.linalg.solve(A.transpose(-2, -1) if use_transpose_bwd else A, grad_u)
            grad_vals_coal = (-(lam @ u.transpose(0, 1)))[rows, cols]
        else:
            M_inv = extra if (use_jacobi and extra.numel() != 0) else None
            if use_transpose_bwd:
                AmvT = lambda x: _spmv_coo_T(rows, cols, vals, x, n)
                lam = _bicgstab(AmvT, grad_u, M_inv, tol=tol, maxiter=maxiter)
            else:
                Amv  = lambda x: _spmv_coo(rows, cols, vals, x, n)
                lam = _pcg(Amv, grad_u, M_inv, tol=tol, maxiter=maxiter)
            grad_vals_coal = -(lam[rows] * u[cols]).sum(dim=1)

        # map back to original a_entries order
        grad_vals = grad_vals_coal[idx_map]

        grad_b = lam.squeeze(1) if single_rhs else lam
        return (None, grad_vals, None, grad_b, None, None, None, None, None, None)


def solve_coo(a_entries: Tensor,
              a_indices: Tensor,
              b: Tensor,
              sym_pos: bool = False,
              tol: float = 1e-8,
              maxiter: int = 2000,
              use_jacobi: bool = True,
              dense_threshold: int = 512,
              dense_density: float = 0.02) -> Tensor:
    n = b.shape[0] if b.dim() > 1 else b.numel()
    # Pass explicit size; Autograd op will validate against b
    return _SolveCOO.apply(a_indices, a_entries, int(n), b, sym_pos, tol, maxiter,
                           use_jacobi, dense_threshold, dense_density)
