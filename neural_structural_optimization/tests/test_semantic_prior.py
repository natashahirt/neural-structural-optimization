# lint as python3
"""Unit tests for the live CLIP/SDS spatial-prior module.

No CLIP checkpoint: providers are callables or the tiny frozen SDS denoiser.
"""

import numpy as np
import torch
from absl.testing import absltest

from neural_structural_optimization.models.loss_semantic_prior import (
    CallableScoreProvider,
    DiffusionSDSProvider,
    FrozenDenoiser,
    SemanticSpatialPrior,
    connectivity_metrics,
    gaussian_blur2d,
    normalize_preference,
    preference_from_score,
    scale_fracs_for_grid,
)
from neural_structural_optimization.models.loss_structural import StructuralLoss
from neural_structural_optimization.models.model_base import VeniceLossAlgebra
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.train.optimizers import AdaptiveAdam_Optimizer

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SCRIPT = _REPO_ROOT / 'script' / 'venice_golden_250214.py'


def _load_golden():
    spec = importlib.util.spec_from_file_location(
        'venice_golden_250214_semantic_prior', _GOLDEN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden = _load_golden()

SMALL_WIDTH = 16
SMALL_HEIGHT = 32
SMALL_INTERVAL = 8


def _blob_score(density: torch.Tensor) -> torch.Tensor:
    """Lower when the top-left quadrant is denser: CLIP-like spatial preference."""
    field = density.reshape(density.shape[-2], density.shape[-1])
    h, w = field.shape
    return -(field[: h // 2, : w // 2].mean())


def _highfreq_score(density: torch.Tensor) -> torch.Tensor:
    field = density.reshape(1, 1, density.shape[-2], density.shape[-1])
    dx = field[..., :, 1:] - field[..., :, :-1]
    return dx.abs().mean()


class PreferenceMathTest(absltest.TestCase):

    def test_preference_is_relu_negative_density_gradient(self):
        density = torch.linspace(0.1, 0.9, 16).reshape(4, 4).requires_grad_(True)
        score = -density[1, 2]
        pref = preference_from_score(density, score, retain_graph=False)
        self.assertFalse(pref.requires_grad)
        self.assertGreater(float(pref[1, 2]), 0.0)
        pref_without = pref.clone()
        pref_without[1, 2] = 0
        self.assertEqual(float(pref_without.max()), 0.0)

    def test_normalize_and_blur_are_in_unit_interval(self):
        pref = torch.tensor([[0.0, 4.0], [1.0, 0.0]])
        normed = normalize_preference(pref)
        self.assertAlmostEqual(float(normed.max()), 1.0)
        blurred = gaussian_blur2d(normed, sigma=1.0)
        self.assertGreaterEqual(float(blurred.min()), 0.0)
        self.assertLessEqual(float(blurred.max()), 1.0 + 1e-5)

    def test_zero_sigma_blur_is_identity(self):
        field = torch.rand(5, 7)
        torch.testing.assert_close(gaussian_blur2d(field, 0.0), field)


class EmaPriorTest(absltest.TestCase):

    def test_ema_and_detachment(self):
        provider = CallableScoreProvider(_blob_score)
        prior = SemanticSpatialPrior(
            provider,
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.5,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        density = torch.ones(8, 8) * 0.3
        density = density + 0.2 * torch.linspace(0, 1, 8).view(8, 1)
        prior.update(density)
        first = prior.maps['global'].clone()
        density2 = density.clone()
        density2[:4, :4] = 0.9
        prior.update(density2)
        second = prior.maps['global']
        expected = 0.5 * first + 0.5 * prior.last_instant['global']
        torch.testing.assert_close(second, expected)
        occ = prior.blended_occupancy(density2)
        self.assertFalse(occ.requires_grad)

    def test_zero_weight_without_auto_ratio_is_inactive(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0},
            weight=0.0,
            auto_ratio=None,
        )
        self.assertFalse(prior.is_active())

    def test_scale_separation_uses_independent_maps(self):
        provider = CallableScoreProvider(
            _blob_score,
            scale_fns={'global': _blob_score, 'storey': _highfreq_score},
        )
        prior = SemanticSpatialPrior(
            provider,
            scale_fracs={'global': 1.0, 'storey': 0.25},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='all',
        )
        density = torch.rand(16, 8)
        prior.update(density)
        self.assertIsNotNone(prior.maps['global'])
        self.assertIsNotNone(prior.maps['storey'])
        self.assertIsNone(prior.maps['member'])
        # Different objectives should not yield identical maps.
        self.assertGreater(
            float((prior.maps['global'] - prior.maps['storey']).abs().max()),
            1e-6)

    def test_hierarchical_curriculum_unlocks_scales_with_resizes(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0, 'storey': 0.25, 'member': 0.0625},
            weight=1.0,
            curriculum='hierarchical',
        )
        self.assertEqual(prior.active_scales(resizes=0, resize_num=2), ('global',))
        self.assertEqual(
            prior.active_scales(resizes=1, resize_num=2), ('global', 'storey'))
        self.assertEqual(
            prior.active_scales(resizes=2, resize_num=2),
            ('global', 'storey', 'member'))

    def test_grid_fracs_match_physical_motif_scale(self):
        fracs = scale_fracs_for_grid(256, 64)
        self.assertAlmostEqual(fracs['global'], 1.0)
        self.assertAlmostEqual(fracs['storey'], 0.25)
        self.assertAlmostEqual(fracs['member'], 0.0625)


class SdsProviderTest(absltest.TestCase):

    def test_sds_noise_bands_differ_by_scale(self):
        provider = DiffusionSDSProvider(FrozenDenoiser(channels=8, seed=0), seed=0)
        self.assertGreater(provider.timestep_bands['global'][0],
                           provider.timestep_bands['storey'][1])
        self.assertGreater(provider.timestep_bands['storey'][0],
                           provider.timestep_bands['member'][1])
        density = torch.rand(1, 8, 4, requires_grad=True)
        scores = provider.score_by_scale(density, ('global', 'member'))
        self.assertEqual(set(scores), {'global', 'member'})
        for value in scores.values():
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(value.requires_grad)

    def test_sds_occupancy_is_detached(self):
        provider = DiffusionSDSProvider(FrozenDenoiser(channels=8, seed=1), seed=1)
        prior = SemanticSpatialPrior(
            provider,
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.5,
            curriculum='global_only',
        )
        density = torch.rand(8, 4)
        prior.update(density)
        occ = prior.blended_occupancy(density)
        self.assertFalse(occ.requires_grad)
        self.assertTrue(torch.all(occ >= 0) and torch.all(occ <= 1))


class ConnectivityMetricsTest(absltest.TestCase):

    def test_column_from_top_to_bottom_is_connected(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[:, 1] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        loads[0, 1] = True
        metrics = connectivity_metrics(density, loads, threshold=0.3)
        self.assertEqual(metrics['top_to_bottom_connected'], 1.0)
        self.assertEqual(metrics['support_to_load_connected'], 1.0)
        self.assertEqual(metrics['floating_mass_fraction'], 0.0)

    def test_floating_blob_is_counted(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[2:4, 1:3] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        loads[0, 0] = True
        metrics = connectivity_metrics(density, loads, threshold=0.3)
        self.assertEqual(metrics['top_to_bottom_connected'], 0.0)
        self.assertGreater(metrics['floating_mass_fraction'], 0.9)


def _tiny_model(clip_loss=None):
    params = StructuralParams(
        problem_name='multistory_building',
        width=SMALL_WIDTH,
        height=SMALL_HEIGHT,
        density=0.3,
        interval=SMALL_INTERVAL,
        filter_width=1.5,
    )
    from neural_structural_optimization.models.model_ada import AdaptivePixelModel
    model = AdaptivePixelModel(
        structural_params=params,
        clip_loss=None,
        seed=0,
        resize_num=0,
        resize_scale=2,
    )
    model.enable_venice_compat_loss(VeniceLossAlgebra(clip_alpha=10.0))
    if clip_loss is not None:
        object.__setattr__(model, 'clip_loss', clip_loss)
    return model


class SemanticPriorModelSeamTest(absltest.TestCase):

    def test_disabled_prior_does_not_change_total_loss(self):
        model = _tiny_model()
        logits = model()
        without = model.get_venice_compat_losses(logits).total_loss
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0},
            weight=0.0,
        )
        model.enable_semantic_prior(prior)
        with_zero = model.get_venice_compat_losses(logits).total_loss
        torch.testing.assert_close(without, with_zero)

    def test_mocked_clip_moves_mass_into_preferred_region(self):
        def clip_fn(logits):
            return (logits ** 2).mean() * 0.0

        model = _tiny_model(clip_loss=clip_fn)
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0},
            weight=50.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        model.enable_semantic_prior(prior)
        logits = model()
        terms = model.get_venice_compat_losses(logits)
        self.assertGreater(float(model._last_semantic_prior_loss), 0.0)
        occ = prior.blended_occupancy(model.get_physical_density(logits))
        self.assertGreater(
            float(occ[..., : SMALL_HEIGHT // 2, : SMALL_WIDTH // 2].mean()),
            float(occ[..., SMALL_HEIGHT // 2 :, SMALL_WIDTH // 2 :].mean()))
        terms.total_loss.backward()
        self.assertIsNotNone(model.z.grad)


class SemanticPriorFeaCountTest(absltest.TestCase):
    """One StructuralLoss forward per AdaptiveAdam step, prior on or off."""

    def _count_forwards(self, enable_prior: bool) -> int:
        calls = {'n': 0}
        original = StructuralLoss.forward

        def wrapped(ctx, logits, env):
            calls['n'] += 1
            return original(ctx, logits, env)

        StructuralLoss.forward = staticmethod(wrapped)
        try:
            def clip_fn(logits):
                return logits.new_tensor(0.4)

            model = _tiny_model(clip_loss=clip_fn)
            if enable_prior:
                model.enable_semantic_prior(SemanticSpatialPrior(
                    CallableScoreProvider(_blob_score),
                    scale_fracs={'global': 1.0},
                    weight=1.0,
                    curriculum='global_only',
                ))
            AdaptiveAdam_Optimizer(
                model,
                max_iterations=3,
                lr=0.2,
                convergence_threshold=0.0,
                max_resize_iteration=50,
            ).optimize()
        finally:
            StructuralLoss.forward = original
        return calls['n']

    def test_prior_does_not_add_fea_forwards(self):
        off = self._count_forwards(False)
        on = self._count_forwards(True)
        self.assertEqual(off, 3)
        self.assertEqual(on, 3)


if __name__ == '__main__':
    absltest.main()
