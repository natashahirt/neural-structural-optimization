# lint as python3
"""Stage 6: sketch occupancy and a spatial mass prior on physical density.

GATE: on a cheap grid, a sketch-guided Adam run has more mass on
occupancy>threshold than a no-sketch control, and mean(density) stays at
volfrac. No Venice quality claim. Occupancy + density panels are written
under script/test_results_pytorch/stage6_sketch/ so a human can look.
"""

# pylint: disable=missing-docstring

from pathlib import Path

import numpy as np
import torch
from absl.testing import absltest

from neural_structural_optimization.experiment import (
    ExperimentConfig,
    SketchConfig,
    venice_250214,
)
from neural_structural_optimization.models.loss_sketch import (
    DEFAULT_OCCUPANCY_THRESHOLD,
    SKETCH_CORPUS,
    SKETCH_DIR,
    apply_sketch_config,
    init_weight_with_occupancy,
    load_site_mask,
    load_sketch_occupancy,
    mass_fraction_on_occupancy,
    save_sketch_visual,
    sketch_mass_prior_loss,
    sketch_motif_descriptor,
    sketch_motif_loss,
    sketch_patch_vocabulary_loss,
)
from neural_structural_optimization.models.model_pixel import PixelModel
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.train.optimizers import Adam_Optimizer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKETCH_ROOT = _REPO_ROOT / SKETCH_DIR
_VISUAL_DIR = _REPO_ROOT / 'script' / 'test_results_pytorch' / 'stage6_sketch'

# Cheap grid: physics is free, divisible, same family as other unit tests.
_WIDTH = 16
_HEIGHT = 32
_INTERVAL = 8
_VOLFRAC = 0.3
_GATE_STEPS = 10
_GATE_LR = 0.2
_GATE_WEIGHT = 80.0

# Cone filter can nudge the mean off volfrac when volume is not enforced last
# (default path, projection off). The gate is "held", not bit-exact.
_VOLUME_ATOL = 0.05


def _params(**overrides) -> StructuralParams:
  kwargs = dict(
      problem_name='multistory_building',
      width=_WIDTH,
      height=_HEIGHT,
      density=_VOLFRAC,
      interval=_INTERVAL,
  )
  kwargs.update(overrides)
  return StructuralParams(**kwargs)


def _left_half_occupancy(height=_HEIGHT, width=_WIDTH) -> np.ndarray:
  occupancy = np.zeros((height, width), dtype=np.float32)
  occupancy[:, : width // 2] = 1.0
  return occupancy


def _mean_density(model: PixelModel) -> float:
  density = model.get_physical_density(model()).detach().cpu().numpy()
  return float(np.mean(density))


def _density_and_mass(model: PixelModel, occupancy: np.ndarray):
  density = model.get_physical_density(model()).detach()
  mass = mass_fraction_on_occupancy(density, occupancy)
  return density.cpu().numpy(), mass


class OccupancyPreprocessTest(absltest.TestCase):
  """Thresholded occupancy of the corpus is inspectable and not a JPEG copy."""

  def test_default_threshold_matches_the_config_knob(self):
    self.assertEqual(DEFAULT_OCCUPANCY_THRESHOLD, SketchConfig().threshold)
    self.assertEqual(DEFAULT_OCCUPANCY_THRESHOLD, 0.40)

  def test_corpus_files_exist(self):
    for name in SKETCH_CORPUS:
      path = _SKETCH_ROOT / name
      self.assertTrue(path.is_file(), msg=f'missing sketch corpus file: {path}')

  def test_occupancy_is_binary_and_smaller_than_the_ink_field(self):
    """Threshold drops paper / faint construction; occupancy is not the JPEG."""
    for name in SKETCH_CORPUS:
      path = _SKETCH_ROOT / name
      occupancy = load_sketch_occupancy(
          path, height=_HEIGHT, width=_WIDTH, threshold=DEFAULT_OCCUPANCY_THRESHOLD)
      unique = set(np.unique(occupancy).tolist())
      self.assertTrue(unique <= {0.0, 1.0}, msg=f'{name} unique={unique}')
      frac = float(occupancy.mean())
      with self.subTest(sketch=name):
        self.assertGreater(frac, 0.05)
        self.assertLess(frac, 0.45)
      # Unthresholded inverted ink is strictly denser: the cut dropped something.
      from PIL import Image
      ink = 1.0 - (np.asarray(
          Image.open(path).convert('L').resize((_WIDTH, _HEIGHT), Image.BILINEAR),
          dtype=np.float64) / 255.0)
      self.assertLess(frac, float((ink >= 0.05).mean()))

  def test_blur_softens_the_binary_map(self):
    occupancy = load_sketch_occupancy(
        _SKETCH_ROOT / '12.jpg',
        height=_HEIGHT,
        width=_WIDTH,
        blur_sigma=1.5,
    )
    interior = occupancy[(occupancy > 0.0) & (occupancy < 1.0)]
    self.assertGreater(interior.size, 0)

  def test_missing_file_raises(self):
    with self.assertRaises(FileNotFoundError):
      load_sketch_occupancy(_SKETCH_ROOT / 'not-a-sketch.jpg', 8, 8)

  def test_saves_occupancy_visuals_for_the_corpus(self):
    panels = []
    for name in SKETCH_CORPUS:
      occupancy = load_sketch_occupancy(
          _SKETCH_ROOT / name, height=64, width=32)
      save_sketch_visual(_VISUAL_DIR / f'{Path(name).stem}_occupancy.png', [occupancy])
      panels.append(occupancy)
    save_sketch_visual(_VISUAL_DIR / 'corpus_occupancy_strip.png', panels)
    self.assertTrue((_VISUAL_DIR / 'corpus_occupancy_strip.png').is_file())


class MassPriorMathTest(absltest.TestCase):

  def test_all_mass_on_occupancy_is_zero_loss(self):
    density = torch.tensor([[0.3, 0.0], [0.3, 0.0]])
    occupancy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = float(sketch_mass_prior_loss(density, occupancy))
    self.assertAlmostEqual(loss, 0.0, places=6)

  def test_all_mass_off_occupancy_is_one(self):
    density = torch.tensor([[0.0, 0.3], [0.0, 0.3]])
    occupancy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = float(sketch_mass_prior_loss(density, occupancy))
    self.assertAlmostEqual(loss, 1.0, places=6)

  def test_uniform_density_loss_is_one_minus_occupancy_mean(self):
    density = torch.ones(4, 8) * 0.3
    occupancy = torch.zeros(4, 8)
    occupancy[:, :4] = 1.0
    loss = float(sketch_mass_prior_loss(density, occupancy))
    self.assertAlmostEqual(loss, 0.5, places=6)

  def test_mass_fraction_metric_matches_the_complement(self):
    density = torch.tensor([[0.2, 0.1], [0.4, 0.3]])
    occupancy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    on_occ = mass_fraction_on_occupancy(density, occupancy)
    loss = float(sketch_mass_prior_loss(density, occupancy))
    self.assertAlmostEqual(on_occ + loss, 1.0, places=6)

  def test_mass_on_load_pixels_is_not_off_template(self):
    occupancy = torch.zeros(4, 8)
    sites = torch.zeros(4, 8)
    sites[0, :] = 1.0
    density_on_floor = torch.zeros(4, 8)
    density_on_floor[0, :] = 0.3
    self.assertAlmostEqual(
        float(sketch_mass_prior_loss(
            density_on_floor, occupancy, load_sites=sites)),
        0.0, places=6)
    density_stray = torch.zeros(4, 8)
    density_stray[2, 4] = 1.0
    self.assertAlmostEqual(
        float(sketch_mass_prior_loss(
            density_stray, occupancy, load_sites=sites)),
        1.0, places=6)

  def test_point_load_allows_one_pixel_not_the_whole_row(self):
    occupancy = torch.zeros(4, 8)
    sites = torch.zeros(4, 8)
    sites[2, 7] = 1.0
    density = torch.zeros(4, 8)
    density[2, :] = 0.1
    loss = float(sketch_mass_prior_loss(density, occupancy, load_sites=sites))
    self.assertAlmostEqual(loss, 0.7 / 0.8, places=6)


class MotifPriorMathTest(absltest.TestCase):

  @staticmethod
  def _vee(height=48, width=64):
    field = torch.zeros(height, width)
    for y in range(6, height - 6):
      left = width // 2 - (y - 6) // 2
      right = width // 2 - 1 + (y - 6) // 2
      field[y, max(0, left - 1):min(width, left + 2)] = 1.0
      field[y, max(0, right - 1):min(width, right + 2)] = 1.0
    return field

  def test_translation_changes_descriptor_far_less_than_columns(self):
    reference = self._vee()
    translated = torch.zeros_like(reference)
    translated[:, 7:] = reference[:, :-7]
    columns = torch.zeros_like(reference)
    columns[:, ::8] = 1.0
    translation_loss = sketch_motif_loss(translated, reference)
    column_loss = sketch_motif_loss(columns, reference)
    self.assertLess(float(translation_loss), 0.01)
    self.assertGreater(float(column_loss), 25.0 * float(translation_loss))

  def test_orientation_change_is_detected(self):
    reference = self._vee()
    horizontal_bars = torch.zeros_like(reference)
    horizontal_bars[::8, :] = 1.0
    self.assertAlmostEqual(
        float(sketch_motif_loss(reference, reference)), 0.0, places=7)
    self.assertGreater(
        float(sketch_motif_loss(horizontal_bars, reference)), 0.05)

  def test_blank_patches_have_zero_descriptor_energy(self):
    blank = torch.zeros(32, 32)
    descriptor = sketch_motif_descriptor(blank, scales=(1, 2, 4))
    np.testing.assert_array_equal(
        descriptor.detach().cpu().numpy(), np.zeros_like(descriptor))

  def test_load_only_row_is_not_part_of_reference_motif(self):
    model = PixelModel(structural_params=_params(), seed=0)
    occupancy = _left_half_occupancy()
    model.enable_sketch_prior(
        occupancy, weight=0.0, motif_weight=1.0, motif_scales=(1, 2))
    density = model.get_physical_density(model())
    occ_t = model._occupancy_on_density(density)
    sites = model._load_sites_on_density(density)
    model_loss = model.get_sketch_motif_loss(model())
    occupancy_only = sketch_motif_loss(density, occ_t, scales=(1, 2))
    allowed_template = sketch_motif_loss(
        density, torch.maximum(occ_t, sites), scales=(1, 2))
    self.assertAlmostEqual(float(model_loss), float(occupancy_only), places=7)
    self.assertNotAlmostEqual(
        float(model_loss), float(allowed_template), places=5)

  def test_gradients_are_finite_and_nonzero(self):
    reference = self._vee(32, 32)
    density = torch.rand(32, 32, requires_grad=True)
    loss = sketch_motif_loss(density, reference, scales=(1, 2, 4))
    loss.backward()
    self.assertIsNotNone(density.grad)
    self.assertTrue(bool(torch.isfinite(density.grad).all()))
    self.assertGreater(float(density.grad.abs().sum()), 0.0)


class PatchVocabularyMathTest(absltest.TestCase):

  def test_translated_sketch_scores_better_than_repeated_columns(self):
    reference = torch.as_tensor(load_sketch_occupancy(
        _SKETCH_ROOT / '12.jpg', height=128, width=64))
    translated = torch.zeros_like(reference)
    translated[:, 5:] = reference[:, :-5]
    columns = torch.zeros_like(reference)
    columns[:, ::8] = 1.0
    translation_loss = sketch_patch_vocabulary_loss(translated, reference)
    column_loss = sketch_patch_vocabulary_loss(columns, reference)
    self.assertLess(float(translation_loss), 0.3)
    self.assertGreater(float(column_loss), 2.0 * float(translation_loss))

  def test_horizontal_rows_score_worse_than_drawing_vocabulary(self):
    reference = torch.as_tensor(load_sketch_occupancy(
        _SKETCH_ROOT / '12.jpg', height=128, width=64))
    rows = torch.zeros_like(reference)
    rows[::8, :] = 1.0
    self.assertGreater(
        float(sketch_patch_vocabulary_loss(rows, reference)), 0.6)

  def test_load_only_pixels_receive_no_patch_gradient(self):
    occupancy = torch.zeros(32, 32)
    occupancy[5:27, 14:18] = 1.0
    sites = torch.zeros_like(occupancy)
    sites[8, :] = 1.0
    density = torch.rand(32, 32, requires_grad=True)
    loss = sketch_patch_vocabulary_loss(
        density, occupancy, load_sites=sites, patch_sizes=(7,))
    loss.backward()
    load_only = sites.bool() & ~occupancy.bool()
    self.assertAlmostEqual(
        float(density.grad[load_only].abs().sum()), 0.0, places=7)

  def test_gradients_are_finite_and_nonzero(self):
    reference = MotifPriorMathTest._vee(32, 32)
    density = torch.rand(32, 32, requires_grad=True)
    loss = sketch_patch_vocabulary_loss(
        density, reference, patch_sizes=(7, 15))
    loss.backward()
    self.assertTrue(bool(torch.isfinite(density.grad).all()))
    self.assertGreater(float(density.grad.abs().sum()), 0.0)

  def test_blank_reference_is_a_differentiable_noop(self):
    density = torch.rand(32, 32, requires_grad=True)
    loss = sketch_patch_vocabulary_loss(density, torch.zeros(32, 32))
    loss.backward()
    self.assertEqual(float(loss), 0.0)
    self.assertEqual(float(density.grad.abs().sum()), 0.0)


class ExperimentConfigSketchTest(absltest.TestCase):

  def test_venice_preset_has_the_prior_off(self):
    cfg = venice_250214()
    self.assertIsNone(cfg.sketch.path)
    self.assertEqual(cfg.sketch.threshold, DEFAULT_OCCUPANCY_THRESHOLD)
    self.assertIsNone(cfg.sketch.weight_end)
    self.assertFalse(cfg.sketch.init_from_occupancy)
    self.assertEqual(cfg.sketch.motif_weight, 0.0)
    self.assertEqual(cfg.sketch.patch_weight, 0.0)
    restored = ExperimentConfig.from_json(cfg.to_json())
    self.assertEqual(restored.sketch, cfg.sketch)

  def test_override_round_trips_init_and_anneal_knobs(self):
    rel = str(SKETCH_DIR / '12.jpg')
    cfg = venice_250214().with_overrides(**{
        'sketch.path': rel,
        'sketch.weight': 4000.0,
        'sketch.weight_end': 400.0,
        'sketch.init_from_occupancy': True,
        'sketch.motif_weight': 1200.0,
        'sketch.motif_weight_end': 400.0,
        'sketch.motif_scales': [1, 2],
        'sketch.patch_weight': 40.0,
        'sketch.patch_weight_end': 20.0,
        'sketch.patch_sizes': [7, 15],
        'sketch.patch_stride': 2,
    })
    self.assertEqual(cfg.sketch.weight, 4000.0)
    self.assertEqual(cfg.sketch.weight_end, 400.0)
    self.assertTrue(cfg.sketch.init_from_occupancy)
    self.assertEqual(cfg.sketch.motif_weight, 1200.0)
    self.assertEqual(cfg.sketch.motif_weight_end, 400.0)
    self.assertEqual(cfg.sketch.motif_scales, (1, 2))
    self.assertEqual(cfg.sketch.patch_weight, 40.0)
    self.assertEqual(cfg.sketch.patch_weight_end, 20.0)
    self.assertEqual(cfg.sketch.patch_sizes, (7, 15))
    cfg.validate()
    restored = ExperimentConfig.from_json(cfg.to_json())
    self.assertEqual(restored, cfg)

  def test_override_round_trips_a_corpus_path(self):
    rel = str(SKETCH_DIR / '12.jpg')
    cfg = venice_250214().with_overrides(**{
        'sketch.path': rel,
        'sketch.weight': 2.5,
    })
    self.assertEqual(cfg.sketch.path, rel)
    self.assertEqual(cfg.sketch.weight, 2.5)
    cfg.validate()
    restored = ExperimentConfig.from_json(cfg.to_json())
    self.assertEqual(restored, cfg)

  def test_missing_sketch_path_fails_validate(self):
    cfg = venice_250214().with_overrides(**{
        'sketch.path': str(SKETCH_DIR / 'does-not-exist.jpg'),
    })
    with self.assertRaisesRegex(ValueError, 'sketch.path'):
      cfg.validate()

  def test_apply_sketch_config_is_a_noop_when_path_is_none(self):
    model = PixelModel(structural_params=_params(), seed=0)
    apply_sketch_config(
        model, SketchConfig(), height=_HEIGHT, width=_WIDTH, repo_root=_REPO_ROOT)
    self.assertIsNone(model.sketch_occupancy_full)

  def test_apply_sketch_config_loads_full_grid_occupancy(self):
    model = PixelModel(structural_params=_params(), seed=0)
    cfg = SketchConfig(path=str(SKETCH_DIR / '12.jpg'), weight=3.0)
    apply_sketch_config(
        model, cfg, height=_HEIGHT, width=_WIDTH, repo_root=_REPO_ROOT)
    self.assertIsNotNone(model.sketch_occupancy_full)
    self.assertEqual(tuple(model.sketch_occupancy_full.shape), (_HEIGHT, _WIDTH))
    self.assertEqual(model.sketch_weight, 3.0)


class LoadSiteMaskTest(absltest.TestCase):

  def test_floor_line_is_one_element_row(self):
    nelx, nely = 8, 4
    forces = np.zeros((nelx + 1, nely + 1, 2))
    forces[:, 0, 1] = -1.0 / nelx
    mask = load_site_mask(forces.ravel(), nely=nely, nelx=nelx)
    self.assertEqual(mask.shape, (nely, nelx))
    np.testing.assert_array_equal(mask[0], np.ones(nelx, dtype=np.float32))
    self.assertEqual(float(mask[1:].sum()), 0.0)

  def test_point_load_is_one_element(self):
    nelx, nely = 8, 4
    forces = np.zeros((nelx + 1, nely + 1, 2))
    forces[nelx, nely // 2, 1] = -1.0
    mask = load_site_mask(forces.ravel(), nely=nely, nelx=nelx)
    self.assertEqual(float(mask.sum()), 1.0)
    self.assertEqual(mask[nely // 2, nelx - 1], 1.0)
    self.assertEqual(float(mask[nely // 2, : nelx - 1].sum()), 0.0)

  def test_ground_node_row_clamps_to_last_element_row(self):
    nelx, nely = 8, 4
    forces = np.zeros((nelx + 1, nely + 1, 2))
    forces[:, nely, 1] = -1.0 / nelx
    mask = load_site_mask(forces, nely=nely, nelx=nelx)
    np.testing.assert_array_equal(mask[-1], np.ones(nelx, dtype=np.float32))
    self.assertEqual(float(mask[:-1].sum()), 0.0)

  def test_multistory_env_marks_floor_rows(self):
    model = PixelModel(structural_params=_params(), seed=0)
    mask = load_site_mask(
        model.env.args['forces'], nely=_HEIGHT, nelx=_WIDTH)
    self.assertTrue(np.all(mask[0] == 1.0))
    self.assertTrue(np.all(mask[_INTERVAL] == 1.0))
    between = mask[1:_INTERVAL, :]
    self.assertEqual(float(between.sum()), 0.0)


class SketchDoesNotChangeANoSketchRunTest(absltest.TestCase):
  """Venice-without-sketch must stay bit-identical: the prior is a no-op."""

  def test_weight_zero_leaves_total_loss_unchanged(self):
    model = PixelModel(structural_params=_params(), seed=0)
    logits = model()
    before = float(model.get_total_loss(logits).detach())
    model.enable_sketch_prior(_left_half_occupancy(), weight=0.0)
    after = float(model.get_total_loss(logits).detach())
    self.assertEqual(before, after)

  def test_nonzero_weight_changes_the_total(self):
    model = PixelModel(structural_params=_params(), seed=0)
    logits = model()
    before = float(model.get_total_loss(logits).detach())
    model.enable_sketch_prior(_left_half_occupancy(), weight=_GATE_WEIGHT)
    after = float(model.get_total_loss(logits).detach())
    self.assertNotEqual(before, after)


class SketchMassConcentratesTest(absltest.TestCase):
  """The Stage 6 gate: more mass on occupancy than a no-sketch control."""

  def test_guided_run_puts_more_mass_on_occupancy_and_holds_volume(self):
    occupancy = _left_half_occupancy()
    params = _params()

    control = PixelModel(structural_params=params, seed=0)
    guided = PixelModel(structural_params=params, seed=0)
    guided.z.data.copy_(control.z.data)
    guided.enable_sketch_prior(occupancy, weight=_GATE_WEIGHT)

    Adam_Optimizer(control, max_iterations=_GATE_STEPS, lr_init=_GATE_LR).optimize()
    Adam_Optimizer(guided, max_iterations=_GATE_STEPS, lr_init=_GATE_LR).optimize()

    control_density, control_mass = _density_and_mass(control, occupancy)
    guided_density, guided_mass = _density_and_mass(guided, occupancy)
    control_mean = float(np.mean(control_density))
    guided_mean = float(np.mean(guided_density))
    sites = load_site_mask(
        guided.env.args['forces'], nely=_HEIGHT, nelx=_WIDTH)
    between = np.maximum(occupancy, 0.0)
    between[sites >= 0.5] = 0.0
    _, control_between = _density_and_mass(control, between)
    _, guided_between = _density_and_mass(guided, between)

    save_sketch_visual(
        _VISUAL_DIR / 'gate_occupancy_guided_control.png',
        [occupancy, np.maximum(occupancy, sites), guided_density, control_density],
    )

    self.assertGreater(
        guided_mass, control_mass,
        msg=(
            f'guided mass-on-occupancy {guided_mass:.4f} was not greater than '
            f'control {control_mass:.4f}'))
    self.assertGreater(
        guided_between, control_between,
        msg=(
            f'guided mass between load sites {guided_between:.4f} was not '
            f'greater than control {control_between:.4f}'))
    np.testing.assert_allclose(guided_mean, _VOLFRAC, atol=_VOLUME_ATOL)
    np.testing.assert_allclose(control_mean, _VOLFRAC, atol=_VOLUME_ATOL)
    np.testing.assert_allclose(guided_mean, control_mean, atol=0.02)

  def test_sketch_only_steps_raise_mass_on_template(self):
    """Even without compliance, the prior redistributes onto occupancy union loads."""
    occupancy = _left_half_occupancy()
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(occupancy, weight=1.0)
    sites = load_site_mask(
        model.env.args['forces'], nely=_HEIGHT, nelx=_WIDTH)
    allowed = np.maximum(occupancy, sites)
    _, mass_before = _density_and_mass(model, allowed)
    optimizer = torch.optim.Adam(model.parameters(), lr=_GATE_LR)
    for _ in range(_GATE_STEPS):
      optimizer.zero_grad(set_to_none=True)
      loss = model.get_sketch_loss(model())
      loss.backward()
      optimizer.step()
    density, mass_after = _density_and_mass(model, allowed)
    save_sketch_visual(
        _VISUAL_DIR / 'sketch_only_occupancy_density.png',
        [occupancy, allowed, density],
    )
    self.assertGreater(mass_after, mass_before)
    np.testing.assert_allclose(float(np.mean(density)), _VOLFRAC, atol=_VOLUME_ATOL)

  def test_get_sketch_loss_uses_env_load_pixels(self):
    occupancy = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    occupancy[:, : _WIDTH // 2] = 1.0
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(occupancy, weight=1.0)
    sites = model._load_sites_on_density(
        model.get_physical_density(model())).detach().cpu().numpy()
    if sites.ndim == 3:
      sites = sites[0]
    self.assertTrue(np.all(sites[0] == 1.0))
    density = torch.zeros(_HEIGHT, _WIDTH)
    density[0, -1] = 1.0
    occ_t = torch.as_tensor(occupancy)
    self.assertAlmostEqual(
        float(sketch_mass_prior_loss(density, occ_t)), 1.0, places=6)
    self.assertAlmostEqual(
        float(sketch_mass_prior_loss(
            density, occ_t, load_sites=torch.as_tensor(sites))),
        0.0, places=6)


class OccupancyInitTest(absltest.TestCase):

  def test_init_writes_occupancy_onto_current_z(self):
    occupancy = _left_half_occupancy()
    model = PixelModel(structural_params=_params(), seed=0)
    init_weight_with_occupancy(model, occupancy)
    z = model.z.detach().cpu().numpy()
    if z.ndim == 3:
      z = z[0]
    np.testing.assert_allclose(z, occupancy, atol=1e-6)

  def test_init_from_occupancy_holds_volfrac_after_physics(self):
    occupancy = _left_half_occupancy()
    model = PixelModel(structural_params=_params(), seed=0)
    apply_sketch_config(
        model,
        SketchConfig(
            path=str(SKETCH_DIR / '12.jpg'),
            weight=1.0,
            init_from_occupancy=True,
        ),
        height=_HEIGHT,
        width=_WIDTH,
        repo_root=_REPO_ROOT,
    )
    density = model.get_physical_density(model()).detach().cpu().numpy()
    np.testing.assert_allclose(float(np.mean(density)), _VOLFRAC, atol=_VOLUME_ATOL)

  def test_apply_without_init_flag_leaves_z_alone(self):
    model = PixelModel(structural_params=_params(), seed=0)
    before = model.z.detach().clone()
    apply_sketch_config(
        model,
        SketchConfig(path=str(SKETCH_DIR / '12.jpg'), weight=1.0),
        height=_HEIGHT,
        width=_WIDTH,
        repo_root=_REPO_ROOT,
    )
    np.testing.assert_allclose(
        model.z.detach().cpu().numpy(), before.cpu().numpy())


class SketchWeightAnnealTest(absltest.TestCase):

  def test_constant_when_weight_end_is_none(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(_left_half_occupancy(), weight=80.0)
    self.assertAlmostEqual(model.sketch_weight_at(step=0, max_iterations=10), 80.0)
    self.assertAlmostEqual(model.sketch_weight_at(step=9, max_iterations=10), 80.0)

  def test_linear_in_step_on_pixel_model(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(
        _left_half_occupancy(), weight=4000.0, weight_end=400.0)
    self.assertAlmostEqual(
        model.sketch_weight_at(step=0, max_iterations=3), 4000.0)
    self.assertAlmostEqual(
        model.sketch_weight_at(step=1, max_iterations=3), 2200.0)
    self.assertAlmostEqual(
        model.sketch_weight_at(step=2, max_iterations=3), 400.0)

  def test_linear_in_adaptive_stage(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(
        _left_half_occupancy(), weight=4000.0, weight_end=400.0)
    model.resize_num = 2
    model.resizes = 0
    self.assertAlmostEqual(model.sketch_weight_at(), 4000.0)
    model.resizes = 1
    self.assertAlmostEqual(model.sketch_weight_at(), 2200.0)
    model.resizes = 2
    self.assertAlmostEqual(model.sketch_weight_at(), 400.0)

  def test_apply_schedule_sets_current_weight(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(
        _left_half_occupancy(), weight=10.0, weight_end=0.0)
    model.apply_sketch_schedule(step=5, max_iterations=11)
    self.assertAlmostEqual(model.sketch_weight, 5.0)

  def test_motif_is_off_coarse_peaks_after_upsample_then_moderates(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(
        _left_half_occupancy(),
        weight=1.0,
        motif_weight=1200.0,
        motif_weight_end=400.0,
    )
    model.resize_num = 2
    model.resizes = 0
    model.apply_sketch_schedule()
    self.assertEqual(model.sketch_motif_weight, 0.0)
    model.resizes = 1
    model.apply_sketch_schedule()
    self.assertEqual(model.sketch_motif_weight, 1200.0)
    model.resizes = 2
    model.apply_sketch_schedule()
    self.assertEqual(model.sketch_motif_weight, 400.0)

  def test_patch_is_off_coarse_peaks_after_upsample_then_moderates(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(
        _left_half_occupancy(),
        weight=1.0,
        patch_weight=80.0,
        patch_weight_end=30.0,
    )
    model.resize_num = 2
    model.resizes = 0
    model.apply_sketch_schedule()
    self.assertEqual(model.sketch_patch_weight, 0.0)
    model.resizes = 1
    model.apply_sketch_schedule()
    self.assertEqual(model.sketch_patch_weight, 80.0)
    model.resizes = 2
    model.apply_sketch_schedule()
    self.assertEqual(model.sketch_patch_weight, 30.0)

  def test_zero_weight_motifs_preserve_add_term_identity(self):
    model = PixelModel(structural_params=_params(), seed=0)
    model.enable_sketch_prior(
        _left_half_occupancy(),
        weight=0.0,
        motif_weight=0.0,
        patch_weight=0.0,
    )
    base_loss = torch.tensor(3.0)
    self.assertIs(model.add_sketch_term(base_loss, model()), base_loss)

  def test_init_and_anneal_gate_beats_control(self):
    occupancy = _left_half_occupancy()
    params = _params()
    control = PixelModel(structural_params=params, seed=0)
    guided = PixelModel(structural_params=params, seed=0)
    guided.enable_sketch_prior(occupancy, weight=80.0, weight_end=20.0)
    init_weight_with_occupancy(guided, occupancy)

    Adam_Optimizer(control, max_iterations=_GATE_STEPS, lr_init=_GATE_LR).optimize()
    Adam_Optimizer(guided, max_iterations=_GATE_STEPS, lr_init=_GATE_LR).optimize()

    _, control_mass = _density_and_mass(control, occupancy)
    guided_density, guided_mass = _density_and_mass(guided, occupancy)
    save_sketch_visual(
        _VISUAL_DIR / 'gate_init_anneal_guided_control.png',
        [occupancy, guided_density],
    )
    self.assertGreater(guided_mass, control_mass)
    np.testing.assert_allclose(
        float(np.mean(guided_density)), _VOLFRAC, atol=_VOLUME_ATOL)


class MotifLayoutScaffoldTest(absltest.TestCase):

  def test_extraction_is_deterministic_and_bounded(self):
    from neural_structural_optimization.models.loss_sketch import (
        motif_layout_scaffold)

    teacher = _left_half_occupancy()
    fracs = (1.0, 0.25, 0.0625)
    a = motif_layout_scaffold(teacher, scale_fracs=fracs)
    b = motif_layout_scaffold(teacher, scale_fracs=fracs)
    np.testing.assert_array_equal(a, b)
    self.assertEqual(a.shape, teacher.shape)
    self.assertGreaterEqual(float(a.min()), 0.0)
    self.assertLessEqual(float(a.max()), 1.0)
    self.assertGreater(float(a.mean()), float(teacher.mean()))

  def test_storey_member_scaffold_is_sparser_than_building_envelope(self):
    from neural_structural_optimization.experiment import (
        physical_motif_scale_fracs)
    from neural_structural_optimization.models.loss_sketch import (
        motif_layout_scaffold)

    teacher = _left_half_occupancy(height=256, width=128)
    building, storey, member = physical_motif_scale_fracs(256, 64)
    fat = motif_layout_scaffold(teacher, scale_fracs=(building, storey, member))
    layout = motif_layout_scaffold(teacher, scale_fracs=(storey, member))
    self.assertLess(float(layout.mean()), float(fat.mean()))
    self.assertLess(float(layout.mean()), 0.85)

  def test_physical_scale_sigmas_match_building_storey_member(self):
    from neural_structural_optimization.experiment import (
        physical_motif_scale_fracs)
    from neural_structural_optimization.models.loss_sketch import (
        motif_layout_envelope_sigmas)

    fracs = physical_motif_scale_fracs(256, 64)
    self.assertEqual(fracs, (1.0, 0.25, 0.0625))
    self.assertEqual(
        motif_layout_envelope_sigmas(256, fracs, 0.25),
        (64.0, 16.0, 4.0))

  def test_broader_scale_envelope_covers_more_area(self):
    from neural_structural_optimization.models.loss_sketch import (
        motif_layout_scaffold)

    teacher = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    teacher[_HEIGHT // 2, _WIDTH // 2] = 1.0
    building = motif_layout_scaffold(
        teacher, scale_fracs=(1.0,), envelope_sigma_frac=0.25)
    member = motif_layout_scaffold(
        teacher, scale_fracs=(0.0625,), envelope_sigma_frac=0.25)
    self.assertGreater(float(building.mean()), float(member.mean()))

  def test_empty_ink_is_rejected(self):
    from neural_structural_optimization.models.loss_sketch import (
        motif_layout_scaffold)

    with self.assertRaisesRegex(ValueError, 'empty'):
      motif_layout_scaffold(
          np.zeros((_HEIGHT, _WIDTH), dtype=np.float32),
          scale_fracs=(1.0,),
          threshold=0.5)

  def test_load_site_union_does_not_punish_loaded_void(self):
    occupancy = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    occupancy[:, : _WIDTH // 2] = 1.0
    load_sites = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    load_sites[-1, -1] = 1.0
    density = torch.zeros(_HEIGHT, _WIDTH)
    density[-1, -1] = 1.0
    loss_without = float(sketch_mass_prior_loss(
        density, torch.as_tensor(occupancy)))
    loss_with = float(sketch_mass_prior_loss(
        density, torch.as_tensor(occupancy),
        load_sites=torch.as_tensor(load_sites)))
    self.assertAlmostEqual(loss_without, 1.0, places=5)
    self.assertAlmostEqual(loss_with, 0.0, places=5)

  def test_apply_layout_is_noop_when_disabled(self):
    from neural_structural_optimization.experiment import MotifLayoutConfig
    from neural_structural_optimization.models.loss_sketch import (
        apply_motif_layout_config)

    model = PixelModel(structural_params=_params(), seed=0)
    before = model.z.detach().clone()
    apply_motif_layout_config(
        model, MotifLayoutConfig(), _left_half_occupancy(),
        scale_fracs=(1.0,))
    self.assertIsNone(model.sketch_occupancy_full)
    torch.testing.assert_close(model.z.detach(), before)

  def test_apply_layout_attaches_scaffold_and_keeps_motif_terms_off(self):
    from neural_structural_optimization.experiment import MotifLayoutConfig
    from neural_structural_optimization.models.loss_sketch import (
        apply_motif_layout_config, motif_layout_scaffold)

    teacher = _left_half_occupancy()
    fracs = (1.0, 0.25)
    expected = motif_layout_scaffold(teacher, scale_fracs=fracs)
    model = PixelModel(structural_params=_params(), seed=0)
    apply_motif_layout_config(
        model,
        MotifLayoutConfig(
            enabled=True, init_from_teacher=False, weight=12.0, weight_end=3.0),
        teacher,
        scale_fracs=fracs,
    )
    self.assertIsNotNone(model.sketch_occupancy_full)
    np.testing.assert_allclose(
        model.sketch_occupancy_full.numpy(), expected, atol=1e-6)
    self.assertEqual(model.sketch_weight_start, 12.0)
    self.assertEqual(model.sketch_weight_end, 3.0)
    self.assertEqual(model.sketch_motif_weight_peak, 0.0)
    self.assertEqual(model.sketch_patch_weight_peak, 0.0)

  def test_init_from_teacher_resamples_without_clamping(self):
    from neural_structural_optimization.models.loss_sketch import (
        init_weight_from_teacher)

    teacher = np.full((_HEIGHT, _WIDTH), 2.5, dtype=np.float32)
    teacher[0, 0] = -4.0
    model = PixelModel(structural_params=_params(), seed=0)
    init_weight_from_teacher(model, teacher)
    z = model.z.detach().cpu().numpy()[0]
    self.assertEqual(z.shape, (_HEIGHT, _WIDTH))
    self.assertGreater(float(z.max()), 1.0)
    self.assertLess(float(z.min()), 0.0)

  def test_venice_preset_layout_is_off(self):
    from neural_structural_optimization.experiment import MotifLayoutConfig

    cfg = venice_250214()
    self.assertFalse(cfg.layout.enabled)
    self.assertEqual(cfg.layout, MotifLayoutConfig())
    restored = ExperimentConfig.from_json(cfg.to_json())
    self.assertEqual(restored.layout, cfg.layout)


if __name__ == '__main__':
  absltest.main()
