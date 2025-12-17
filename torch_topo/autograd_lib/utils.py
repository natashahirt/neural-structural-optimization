"""
Utility functions for neural structural optimization.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor
Device = torch.device


def get_optimal_device() -> Device:
    """Get the best available device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inverse_permutation(indices: Tensor) -> Tensor:
    """
    Return inverse permutation such that inverse_perm[indices[k]] = k 
    """
    if indices.dtype != torch.long:
        indices = indices.long()
    inverse_perm = torch.empty_like(indices)
    inverse_perm[indices] = torch.arange(indices.numel(), device=indices.device, dtype=torch.long)
    return inverse_perm


def scatter1d(nonzero_values: Tensor, nonzero_indices: Tensor, array_len: int) -> Tensor:
    """
    L-length vector with given values at given indices, zeros elsewhere
    """
    out = nonzero_values.new_zeros(array_len, dtype=nonzero_values.dtype)
    out.scatter_(0, nonzero_indices.long(), nonzero_values)
    return out


def normalize_sparse_indices(indices: Tensor) -> Tensor:
    """
    Accept [2,K] or [K,2]; return [2,K] long.
    Normalizes sparse matrix indices to consistent format.
    """
    if indices.dim() != 2:
        raise ValueError("indices must be 2D")
    if indices.shape[0] == 2:
        out = indices
    elif indices.shape[1] == 2:
        out = indices.t()
    else:
        raise ValueError("indices must be shape (2, nnz) or (nnz, 2)")
    return out.long()
