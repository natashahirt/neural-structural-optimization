"""Structural loss function that bridges PyTorch models with NumPy/HIPS-autograd physics."""

import math
from typing import Any, Dict, Tuple

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
            dtype=ctx.dtype, 
            device=ctx.device
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: compute gradients using autograd and map back to PyTorch."""
        (logits_cpu,) = ctx.saved_tensors
        env = ctx.env

        # Convert to NumPy double precision (shape: batch, H, W or similar)
        x_np = logits_cpu.double().numpy()

        # If the environment provides an analytical gradient (pyFANTOM-backed),
        # prefer it to avoid HIPS autograd through external solvers.
        g_np = None
        if hasattr(env, "objective_grad"):
            try:
                # Support batch size 1 (typical path)
                if x_np.ndim >= 2:
                    batch_dim = x_np.shape[0]
                    g_np = np.zeros_like(x_np, dtype=np.float64)
                    go = grad_output.detach().cpu().to(torch.float64).numpy()
                    for i in range(batch_dim):
                        gi = env.objective_grad(x_np[i], volume_constraint=True)
                        if isinstance(gi, np.ndarray) and gi.shape != x_np[i].shape:
                            gi = np.asarray(gi).reshape(x_np[i].shape)
                        g_np[i] = np.asarray(gi, dtype=np.float64) * float(go[i])
                else:
                    gi = env.objective_grad(x_np, volume_constraint=True)
                    g_np = np.asarray(gi, dtype=np.float64) * float(grad_output.detach().cpu().to(torch.float64).item())
            except Exception:
                g_np = None  # fall back to autograd route below

        if g_np is None:
            # Fallback: use HIPS autograd through the legacy autograd physics
            def scalar_objective(x_arr: np.ndarray) -> float:
                l = batched_topo_loss(x_arr, [env])  # -> (batch,)
                go = grad_output.detach().cpu().to(torch.float64).numpy()
                return (l * go).sum()

            import autograd  # type: ignore
            g_np = autograd.grad(scalar_objective)(x_np)

        # Map back to torch, match original dtype & device
        g = torch.from_numpy(g_np).to(ctx.device).to(ctx.dtype)
        return g, None  # no grad for env