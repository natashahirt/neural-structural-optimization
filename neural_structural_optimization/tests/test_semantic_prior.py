# lint as python3
"""Unit tests for the live CLIP/SDS spatial-prior module.

No CLIP checkpoint: providers are callables or the tiny frozen SDS denoiser.
"""

import importlib.util
import inspect
from pathlib import Path
import tempfile

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
    heaviside_projection,
    normalize_preference,
    preference_from_score,
    projected_density_view,
    report_design_metrics,
    save_design_arrays,
    scale_fracs_for_grid,
)
from neural_structural_optimization.models.loss_sketch import (
    mass_fraction_on_occupancy,
    scaffold_spatial_mass_loss,
)
from neural_structural_optimization.models.loss_structural import StructuralLoss
from neural_structural_optimization.models.model_base import VeniceLossAlgebra
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.train.optimizers import AdaptiveAdam_Optimizer

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


def _prefers_material(density: torch.Tensor) -> torch.Tensor:
    """Uniformly rewards material, so preference reflects only the view's slope."""
    return -density.mean()


class ProjectionTest(absltest.TestCase):
    """The sculptural view: CLIP must commit material, not lay down gray."""

    def test_zero_beta_is_identity(self):
        field = torch.rand(6, 5)
        torch.testing.assert_close(heaviside_projection(field, 0.0), field)
        torch.testing.assert_close(
            projected_density_view(field, beta=0.0, filter_sigma=0.0), field)

    def test_projection_pushes_density_toward_zero_one(self):
        field = torch.tensor([0.05, 0.5, 0.95])
        projected = heaviside_projection(field, beta=8.0, eta=0.5)
        self.assertLess(float(projected[0]), 0.02)
        self.assertAlmostEqual(float(projected[1]), 0.5, places=5)
        self.assertGreater(float(projected[2]), 0.98)

    def test_faint_density_earns_far_less_preference_than_committed(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
            projection_beta=8.0,
            projection_filter_sigma=0.0,
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        prior.update(density)
        pref = prior.maps['global']
        self.assertGreater(float(pref[:, 4:].mean()), 20 * float(pref[:, :4].mean()))

    def test_without_projection_faint_and_committed_score_alike(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        prior.update(density)
        pref = prior.maps['global']
        torch.testing.assert_close(pref[:, :4].mean(), pref[:, 4:].mean())

    def test_filter_erases_features_below_the_minimum_size(self):
        field = torch.zeros(16, 16)
        field[:, 1:5] = 1.0  # wider than the filter
        field[:, 12] = 1.0  # a single-cell stroke
        view = projected_density_view(field, beta=8.0, filter_sigma=2.0)
        self.assertGreater(float(view[8, 3]), 0.5)
        self.assertLess(float(view[8, 12]), 0.1)

    def test_projection_is_confined_to_the_sculptural_scales(self):
        kwargs = dict(
            scale_fracs={'global': 1.0, 'storey': 0.25},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='all',
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        baseline = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material), **kwargs)
        sculpted = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            projection_beta=8.0,
            projection_scales=('global',),
            **kwargs,
        )
        baseline.update(density)
        sculpted.update(density)
        self.assertTrue(sculpted.projects('global'))
        self.assertFalse(sculpted.projects('storey'))
        self.assertFalse(sculpted.projects('member'))
        # The detail scale is untouched; only the sculptural scale changes.
        torch.testing.assert_close(sculpted.maps['storey'], baseline.maps['storey'])
        self.assertGreater(
            float((sculpted.maps['global'] - baseline.maps['global']).abs().max()),
            1e-3)

    def test_ink_fraction_separates_faint_from_committed_preference(self):
        density = torch.tensor([[0.05, 0.9]])
        faint = torch.tensor([[1.0, 0.0]])
        committed = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(
            SemanticSpatialPrior._ink_fraction(density, faint), 1.0)
        self.assertAlmostEqual(
            SemanticSpatialPrior._ink_fraction(density, committed), 0.0)
        self.assertEqual(
            SemanticSpatialPrior._ink_fraction(density, torch.zeros(1, 2)), 0.0)

    def test_projection_reduces_the_reported_ink_fraction(self):
        kwargs = dict(
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        baseline = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material), **kwargs)
        sculpted = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            projection_beta=8.0, **kwargs)
        baseline.update(density)
        sculpted.update(density)
        self.assertLess(
            sculpted.last_metrics['preference_ink_fraction'],
            baseline.last_metrics['preference_ink_fraction'])

    def test_invalid_projection_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            SemanticSpatialPrior(
                CallableScoreProvider(_blob_score),
                scale_fracs={'global': 1.0},
                projection_eta=0.0,
            )
        with self.assertRaises(ValueError):
            SemanticSpatialPrior(
                CallableScoreProvider(_blob_score),
                scale_fracs={'global': 1.0},
                projection_scales=('facade',),
            )


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


class DesignReportMetricsTest(absltest.TestCase):
    """Shared summary helper relocates validity; scaffold keys are always present."""

    def test_validity_matches_connectivity_metrics_plus_mean(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[:, 1] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        loads[0, 1] = True
        expected = connectivity_metrics(density, loads, threshold=0.3)
        expected['mean_physical_density'] = float(density.mean())
        report = report_design_metrics(density, loads)
        self.assertEqual(report['validity'], expected)
        self.assertEqual(
            report['mean_physical_density'], expected['mean_physical_density'])
        self.assertIsNone(report['spatial_mass_loss'])
        self.assertIsNone(report['mass_on_scaffold'])
        self.assertIsNone(report['clip_loss'])
        self.assertIsNone(report['clip_loss_raw'])

    def test_scaffold_metrics_match_the_sketch_helpers(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[:, 2:] = 1.0
        scaffold = np.zeros((8, 4), dtype=np.float32)
        scaffold[:, :2] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        report = report_design_metrics(density, loads, scaffold=scaffold)
        self.assertEqual(
            report['mass_on_scaffold'],
            mass_fraction_on_occupancy(density, scaffold))
        self.assertEqual(
            report['spatial_mass_loss'],
            scaffold_spatial_mass_loss(density, scaffold, loads))
        self.assertIn('component_count', report['validity'])
        self.assertIn('support_to_load_connected', report['validity'])

    def test_trajectory_clip_keys_are_relocated_not_recomputed(self):
        density = np.ones((2, 2), dtype=np.float32) * 0.3
        loads = np.zeros_like(density, dtype=bool)

        class _FakeDs(dict):
            pass

        ds = _FakeDs()
        ds['clip_loss'] = [1.0, 2.5]
        ds['clip_loss_raw'] = [0.4, 0.37]
        report = report_design_metrics(density, loads, ds=ds)
        self.assertEqual(report['clip_loss'], 2.5)
        self.assertEqual(report['clip_loss_raw'], 0.37)

    def test_npy_round_trip_is_bit_identical_and_unclipped(self):
        density = np.array([[0.25, 0.35], [1.2, -0.1]], dtype=np.float32)
        raw = np.array([[-11.5, 13.1], [0.0, 0.9]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_design_arrays(tmp, density, raw=raw)
            loaded_density = np.load(paths['physical_density'])
            loaded_raw = np.load(paths['final_design_raw'])
        np.testing.assert_array_equal(loaded_density, density)
        np.testing.assert_array_equal(loaded_raw, raw)

    def test_save_without_raw_does_not_write_raw_file(self):
        density = np.array([[0.25, 0.35]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = save_design_arrays(directory, density)
            self.assertIn('physical_density', paths)
            self.assertNotIn('final_design_raw', paths)
            self.assertFalse((directory / 'final_design_raw.npy').exists())
            loaded = np.load(paths['physical_density'])
        np.testing.assert_array_equal(loaded, density)
        self.assertEqual(loaded.dtype, np.float64)

    def test_saliency_and_dream_reports_share_the_same_keys(self):
        density = np.ones((4, 4), dtype=np.float32) * 0.3
        loads = np.zeros_like(density, dtype=bool)
        scaffold = np.ones_like(density)
        class _FakeDs(dict):
            pass
        ds = _FakeDs()
        ds['clip_loss'] = [0.9]
        ds['clip_loss_raw'] = [0.4]
        saliency = report_design_metrics(density, loads, ds=ds)
        dream = report_design_metrics(
            density, loads, scaffold=scaffold, ds=ds)
        self.assertEqual(set(saliency), set(dream))
        self.assertEqual(
            set(saliency),
            {
                'validity',
                'mean_physical_density',
                'clip_loss',
                'clip_loss_raw',
                'mass_on_scaffold',
                'spatial_mass_loss',
            })
        self.assertIsNone(saliency['mass_on_scaffold'])
        self.assertIsNotNone(dream['mass_on_scaffold'])


_SALIENCY_MODULE = None
_DREAM_MODULE = None
_STAGE6_MODULE = None


def _load_saliency_script():
    global _SALIENCY_MODULE
    if _SALIENCY_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            'clip_saliency_compliance_test',
            _REPO_ROOT / 'script' / 'clip_saliency_compliance.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SALIENCY_MODULE = module
    return _SALIENCY_MODULE


def _load_dream_script():
    global _DREAM_MODULE
    if _DREAM_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            'clip_dream_layout_test',
            _REPO_ROOT / 'script' / 'clip_dream_layout.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DREAM_MODULE = module
    return _DREAM_MODULE


def _load_stage6_script():
    global _STAGE6_MODULE
    if _STAGE6_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            'stage6_sketch_on_golden_test',
            _REPO_ROOT / 'script' / 'stage6_sketch_on_golden.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _STAGE6_MODULE = module
    return _STAGE6_MODULE


class FieldExportScaleTest(absltest.TestCase):
    """Absolute [0, 1] density export vs per-panel min-max normalisation."""

    def test_narrow_band_density_is_mid_gray_when_absolute(self):
        module = _load_saliency_script()
        field = np.array([[0.25, 0.35], [0.30, 0.28]], dtype=np.float64)
        absolute = module._field_to_uint8(field, scale='absolute')
        normalized = module._field_to_uint8(field, scale='normalized')
        expected_abs = (255.0 * (1.0 - field)).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(absolute, expected_abs)
        self.assertEqual(int(normalized[0, 0]), 255)
        self.assertEqual(int(normalized[0, 1]), 0)
        self.assertGreater(int(absolute.min()), 140)
        self.assertLess(int(absolute.max()), 210)

    def test_constant_field_is_white_normalized_and_gray_absolute(self):
        module = _load_saliency_script()
        field = np.full((4, 4), 0.3, dtype=np.float64)
        absolute = module._field_to_uint8(field, scale='absolute')
        normalized = module._field_to_uint8(field, scale='normalized')
        np.testing.assert_array_equal(normalized, np.full((4, 4), 255, dtype=np.uint8))
        expected = (255.0 * (1.0 - field)).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(absolute, expected)

    def test_png_round_trip_uses_the_requested_scale(self):
        module = _load_saliency_script()
        field = np.array([[0.25, 0.35], [0.30, 0.28]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            abs_path = Path(tmp) / 'abs.png'
            norm_path = Path(tmp) / 'norm.png'
            module._save_field(abs_path, field, scale='absolute')
            module._save_field(norm_path, field, scale='normalized')
            from PIL import Image
            abs_pixels = np.asarray(Image.open(abs_path))
            norm_pixels = np.asarray(Image.open(norm_path))
        np.testing.assert_array_equal(
            abs_pixels, module._field_to_uint8(field, scale='absolute'))
        np.testing.assert_array_equal(
            norm_pixels, module._field_to_uint8(field, scale='normalized'))
        self.assertNotEqual(int(abs_pixels[0, 0]), int(norm_pixels[0, 0]))

    def test_unknown_scale_is_rejected(self):
        module = _load_saliency_script()
        with self.assertRaises(ValueError):
            module._field_to_uint8(np.ones((2, 2)), scale='minmax')

    def test_png_clips_out_of_range_values_npy_does_not(self):
        """The scientific artifact is the array; the PNG is a [0, 1] preview."""
        module = _load_saliency_script()
        field = np.array([[-0.1, 1.2], [0.0, 1.0]], dtype=np.float64)
        png = module._field_to_uint8(field, scale='absolute')
        np.testing.assert_array_equal(
            png, np.array([[255, 0], [255, 0]], dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            loaded = np.load(save_design_arrays(tmp, field)['physical_density'])
        np.testing.assert_array_equal(loaded, field)

    def test_batched_field_squeezes_to_2d_before_png(self):
        module = _load_saliency_script()
        field = np.array([[[0.25, 0.35], [0.30, 0.28]]], dtype=np.float64)
        pixels = module._field_to_uint8(field, scale='absolute')
        self.assertEqual(pixels.shape, (2, 2))
        expected = (255.0 * (1.0 - field[0])).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(pixels, expected)

    def test_both_pipelines_bind_the_shared_export_helpers(self):
        saliency = _load_saliency_script()
        dream = _load_dream_script()
        self.assertIs(saliency.save_design_arrays, save_design_arrays)
        self.assertIs(dream.save_design_arrays, save_design_arrays)
        self.assertIs(saliency.report_design_metrics, report_design_metrics)
        self.assertIs(dream.report_design_metrics, report_design_metrics)

    def test_occupancy_like_band_is_not_full_contrast_when_absolute(self):
        """[0.04, 0.07] occupancy must stay a faint ink wash, not min-max 0-255."""
        module = _load_saliency_script()
        field = np.array([[0.0477, 0.069], [0.04, 0.07]], dtype=np.float64)
        absolute = module._field_to_uint8(field, scale='absolute')
        normalized = module._field_to_uint8(field, scale='normalized')
        expected = (255.0 * (1.0 - field)).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(absolute, expected)
        self.assertEqual(int(normalized.min()), 0)
        self.assertEqual(int(normalized.max()), 255)
        # inverted: material is black, so a ~0.05 occupancy is near-white paper
        self.assertGreater(int(absolute.min()), 230)
        self.assertLess(int(absolute.max()) - int(absolute.min()), 20)

    def test_occupancy_panels_use_absolute_scale(self):
        src = inspect.getsource(_load_saliency_script().run_arm)
        self.assertIn(
            "semantic_occupancy.png', occupancy, scale='absolute'", src)
        self.assertIn("('CLIP/SDS prior', occupancy, 'absolute')", src)
        self.assertIn(
            "compliance_sensitivity.png', sensitivity, scale='normalized'",
            src)
        self.assertIn(
            "semantic_preference.png', preference, scale='normalized'", src)
        self.assertNotIn(
            "semantic_occupancy.png', occupancy, scale='normalized'", src)

    def test_run_arm_passes_blended_occupancy_as_scaffold(self):
        src = inspect.getsource(_load_saliency_script().run_arm)
        self.assertIn('scaffold=occupancy', src)
        occupancy_at = src.index('occupancy = prior.blended_occupancy')
        report_at = src.index('report_design_metrics')
        self.assertLess(occupancy_at, report_at)

    def test_scaffold_none_stays_null_not_zero(self):
        density = np.ones((4, 4), dtype=np.float32) * 0.3
        loads = np.zeros_like(density, dtype=bool)
        report = report_design_metrics(density, loads, scaffold=None)
        self.assertIsNone(report['mass_on_scaffold'])
        self.assertIsNone(report['spatial_mass_loss'])
        self.assertIsNot(report['mass_on_scaffold'], 0.0)

    def test_dream_scaffold_png_is_absolute_not_rank_stretched(self):
        dream = _load_dream_script()
        field = np.array([[0.25, 0.35], [0.30, 0.28]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            abs_path = Path(tmp) / 'scaffold.png'
            rank_path = Path(tmp) / 'rank.png'
            dream._save_field_png(abs_path, field, rank=False)
            dream._save_field_png(rank_path, field, rank=True)
            from PIL import Image
            absolute = np.asarray(Image.open(abs_path))
            ranked = np.asarray(Image.open(rank_path))
        expected = (255.0 * (1.0 - field)).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(absolute, expected)
        self.assertGreater(int(absolute.min()), 140)
        self.assertLess(int(absolute.max()), 210)
        self.assertEqual(int(ranked[0, 0]), 255)
        self.assertEqual(int(ranked[0, 1]), 0)

    def test_default_arms_omit_sds_and_folded_venation(self):
        module = _load_saliency_script()
        names = {part.strip() for part in module.DEFAULT_ARMS.split(',')}
        self.assertNotIn('sds_prior', names)
        self.assertNotIn('venation', names)
        self.assertNotIn('venation_projected', names)
        self.assertIn('clip_prior', names)
        self.assertIn('projected', names)

    def test_folded_venation_arms_raise(self):
        module = _load_saliency_script()
        with self.assertRaisesRegex(ValueError, 'folded arm names'):
            module.main(['--arms', 'venation', '--skip-full'])

    def test_full_grid_config_is_shared(self):
        module = _load_saliency_script()
        a = module.full_grid_config('human skull')
        b = module.full_grid_config('human skull')
        self.assertEqual(a, b)
        self.assertEqual(a.width, 128)
        self.assertEqual(a.height, 256)
        self.assertEqual(a.interval, 64)
        self.assertEqual(a.resize_num, 2)
        self.assertTrue(a.neutral_init)
        self.assertEqual(a.motif_scale_fracs, ())


class Stage6ResultLayoutTest(absltest.TestCase):
    """Stage 6 writes under SUCCESS_sketch_to_structure and saves density .npy."""

    def test_default_output_dir_resolves_nested_success_tree(self):
        module = _load_stage6_script()
        out = module._output_dir_for('12.jpg')
        run = out / 'sketch_run.png'
        self.assertEqual(
            out.parts[-2:], ('SUCCESS_sketch_to_structure', 'stage6_sketch12_init_anneal'))
        self.assertTrue(run.is_file(), msg=f'missing {run}')
        self.assertEqual(
            module._stage6_baseline_run('12.jpg'), run)
        motif = module._output_dir_for('12.jpg', motif_scale=True)
        self.assertEqual(motif.name, 'clip_motif_scale_sketch12')
        self.assertNotIn('SUCCESS_sketch_to_structure', motif.parts)

    def test_corpus_strip_reads_success_sketch_run(self):
        module = _load_stage6_script()
        with tempfile.TemporaryDirectory() as tmp:
            strip = module.save_corpus_strip(('12.jpg',), Path(tmp))
            self.assertTrue(strip.is_file())
            self.assertEqual(strip.name, 'corpus_comparison.png')

    def test_corpus_strip_raises_when_sketch_run_is_missing(self):
        module = _load_stage6_script()
        original = module._output_dir_for
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / 'missing'
            empty.mkdir()
            module._output_dir_for = lambda name, motif_scale=False: empty
            try:
                with self.assertRaises(FileNotFoundError) as ctx:
                    module.save_corpus_strip(('12.jpg',), Path(tmp) / 'out')
            finally:
                module._output_dir_for = original
        self.assertIn('sketch_run.png', str(ctx.exception))

    def test_stage6_saves_unclipped_density_array(self):
        module = _load_stage6_script()
        self.assertIs(module.save_design_arrays, save_design_arrays)
        src = inspect.getsource(module.run_sketch_on_golden)
        self.assertIn('save_design_arrays(', src)
        self.assertIn('output_dir, density', src)
        self.assertIn('_ink_black(np.clip(density, 0.0, 1.0), size)', src)
        self.assertNotIn('save_design_arrays(output_dir, np.clip', src)


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
