"""Structural loss function that bridges PyTorch models with NumPy/HIPS-autograd physics."""

import numpy as np
import torch
from typing import Any

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
