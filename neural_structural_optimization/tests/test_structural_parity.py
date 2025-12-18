"""Parity check between PyTorch structural loss and NumPy/autograd physics."""

import numpy as np
import torch

from neural_structural_optimization.models.loss_structural import torch_structural_loss
from neural_structural_optimization.structural import physics, api as topo_api
from neural_structural_optimization.structural.problems import StructuralParams


def _numpy_objective(logits_np: np.ndarray, env) -> float:
    """Compute baseline objective using the NumPy/autograd path."""
    x2d = logits_np.reshape(env.args["nely"], env.args["nelx"])
    return physics.objective(
        x2d,
        env.ke,
        env.args,
        volume_constraint=True,
        cone_filter=True,
    )


def test_torch_matches_numpy_objective():
    # Small problem for a fast check
    params = StructuralParams(problem_name="mbb_beam", width=12, height=6, density=0.4)
    problem = params.get_problem()
    args = topo_api.specified_task(problem)
    env = topo_api.Environment(args)

    torch.manual_seed(0)
    batch = 3
    logits = torch.randn(batch, args["nely"], args["nelx"], dtype=torch.float64)

    torch_loss = torch_structural_loss(
        logits,
        env,
        volume_constraint=True,
        cone_filter=True,
        cg_tol=1e-8,
        cg_max_iter=400,
    ).detach().cpu().numpy()

    numpy_loss = np.array([_numpy_objective(logits[i].cpu().numpy(), env) for i in range(batch)])

    np.testing.assert_allclose(torch_loss, numpy_loss, rtol=1e-4, atol=1e-6)

