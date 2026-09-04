# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity harness for the Venice ``250214_skeleton_loss_test_balanced_dynamic``
golden run.

The configuration under test lives in `script/venice_golden_250214.py`; this
file only compares its output against the reference. The reference trajectory is
transcribed below rather than read from the Venice checkout, so the baseline
travels with this repository and cannot quietly change meaning when a sibling
directory moves.

WHAT IS COMPARED, AND WHY IT IS THE WHOLE CURVE
-----------------------------------------------
Venice logged four fields every five iterations, at iterations 4, 9, ... 124.
Comparing only the endpoint would let a run that took a completely different
path pass on one lucky number, so the assertions run over every recorded point
and additionally over the distribution of per-point errors.

WHY THE TOLERANCE IS STATISTICAL
--------------------------------
The CLIP term is a Monte-Carlo estimate: each evaluation draws 32 random crops
through Kornia's augmentation stack. Both repositories seed torch before the
loop, and Kornia draws from torch's global generator, so each run is
reproducible against ITSELF -- but the number of draws everything upstream
consumes differs between the two codebases, so Venice's crop sequence and this
one's are different samples from the same distribution. Bit-exact replay is
therefore unavailable in principle, not merely unachieved.

The size of that sampling noise was measured directly: 30 independent
evaluations of the raw CLIP term on the FIXED initial design (32x64, 32 crops,
prompt "skeletons") gave a mean of 0.4444 with a relative standard deviation of
0.13% and a full range of 0.60%. That is the per-evaluation floor. The
trajectory tolerances below are larger because the noise enters the GRADIENT at
every one of ~124 steps, so two runs drift apart rather than tracking each other
with a fixed error, and because of the one deliberate modelling difference noted
under `fix_right_wall` in `structural.problems.multistory_building`: this
repository constrains X on the right wall and Venice does not, worth about 0.07%
on compliance, because without it the stiffness matrix is singular and CHOLMOD
refuses the solve outright.

WHAT THE REPRODUCTION ACTUALLY ACHIEVES
---------------------------------------
Better than the tolerances require. A full replay converged at step 122 against
the reference's 124, fired both upsamples at the steps the reference log
implies, and tracked the logged curve to a median relative error under 1% on
every term, ending within 0.6% of the reference's final total. The measured
error distribution and the tolerances derived from it are tabulated at
`TRAJECTORY_RTOL`.

PRECISION: float32 IS THE PARITY-CORRECT CHOICE
-----------------------------------------------
`AdaptivePixelModel` stores its design parameter as float32, and that is
REQUIRED, not a defect to be repaired. Venice's golden run initialized its
parameter from `transforms.ToTensor()`, which emits float32, and its upsample
preserves dtype, so the reference run's parameter was float32 from the first
step to the last. Widening this repository's parameter to float64 would change
how Adam accumulates over 124 steps and would therefore be a divergence from
the reference, not a refinement of it.

Two follow-on facts, checked in `VeniceGoldenPrecisionTest`:

* Storing the field in float32 costs nothing in the physics. The autograd
  backend promotes a float32 input to float64 internally, and the objective on
  the float32 field is bit-identical to the objective on the same field widened
  to float64. Venice fed float32 straight in; this repository casts to float64
  first; the two agree exactly.
* What this repository does lose is the returned scalar, which
  `StructuralLoss.forward` casts back to the input dtype -- so compliance is
  reported to float32 resolution here and to float64 in Venice. That is a
  relative difference of order 1e-7, which reaches the trajectory only through
  the undetached CLIP weight, and it is four orders of magnitude below the
  sampling noise above. It is not the binding constraint on the tolerance and
  no tolerance here is set tighter than float32 resolution.

RUNNING THE FULL REPRODUCTION
-----------------------------
The default suite runs only cheap checks plus a heavily reduced smoke run. The
full 128x256 reproduction takes roughly a quarter of an hour and is opt-in::

    VENICE_PARITY_FULL=1 python -m pytest \\
        neural_structural_optimization/tests/test_venice_parity.py -q

To eyeball a replay instead of asserting on it, run the configuration directly::

    PYTHONPATH="$PWD" python script/venice_golden_250214.py
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from absl.testing import absltest

from neural_structural_optimization.models.loss_clip import VENICE_CLIP_PRESET
from neural_structural_optimization.models.model_base import (
    VeniceLossAlgebra,
    venice_compat_total_loss,
)
from neural_structural_optimization.models.utils import batched_topo_loss

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SCRIPT = _REPO_ROOT / 'script' / 'venice_golden_250214.py'


def _load_golden_config_module():
  """Import the golden configuration from `script/`, which is not a package."""
  spec = importlib.util.spec_from_file_location(
      'venice_golden_250214', _GOLDEN_SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


golden = _load_golden_config_module()


# ---------------------------------------------------------------------------
# The reference trajectory, transcribed from
# semantopology_venice/results/250214_skeleton_loss_test_balanced_dynamic/
#   2_final/00-log-P_multistory_building-M_Ada-T_skeletons-W_128-H_256-V_0.30
#   -LR_0.20-CW_10-ID_01-CompL_74.00-ClipL_0.38-VA_0.31-balanced_dynamic.json
#
# The log stores each series with a leading null (its initial placeholder);
# that placeholder is dropped here, so index i of every list below belongs to
# GOLDEN_STEPS[i]. Venice's step numbers count from 1 -- its iteration counter
# is initialized to 1 rather than 0 -- so the reference's step `s` is this
# repository's zero-based step index `s - 1`.
# ---------------------------------------------------------------------------

GOLDEN_STEPS = list(range(4, 129, 5))

GOLDEN_COMPLIANCE = [
    358.64767390153054, 293.5146195319975, 258.57899130944685,
    228.9510952766208, 207.8366641584774, 193.12165769823946,
    180.4573191757829, 169.54276973414483, 160.60711681121973,
    153.35279067887365, 121.05511933061614, 107.50451940496767,
    100.7569785977627, 96.96923813285018, 87.00399638578422,
    81.27430316930085, 78.94940769382053, 77.59148528809129,
    76.61759937762375, 75.88161676532154, 75.31118594268384,
    74.87061034009182, 74.5371592937928, 74.25581248836946,
    73.99691670938864,
]

GOLDEN_CLIP_LOSS = [
    1544.453771571255, 1221.4890916946395, 1073.2670897018152,
    952.685352584663, 861.6376954741444, 786.2537934340683,
    731.2862607833897, 685.9960586422206, 646.0890217116614,
    620.271877328726, 481.57259573345516, 426.2999442059115,
    396.6629623370226, 381.4787875971676, 338.05313200936234,
    317.07334242312476, 304.85257535538574, 301.46574768082877,
    293.47832157824314, 292.7495844336926, 284.1602326797789,
    285.0874766396376, 281.9547781291462, 281.86193257285,
    283.3703603894398,
]

GOLDEN_CLIP_LOSS_RAW = [
    0.4306325912475586, 0.4161595404148102, 0.4150635302066803,
    0.4161086678504944, 0.41457444429397583, 0.4071287512779236,
    0.40524056553840637, 0.4046153426170349, 0.40227919816970825,
    0.40447381138801575, 0.3978126645088196, 0.39654141664505005,
    0.3936828672885895, 0.3934018611907959, 0.38854897022247314,
    0.39012742042541504, 0.38613662123680115, 0.3885294198989868,
    0.38304296135902405, 0.38579776883125305, 0.3773147761821747,
    0.38077354431152344, 0.3782741129398346, 0.37958231568336487,
    0.3829488754272461,
]

GOLDEN_TOTAL = [
    1903.5320780640332, 1515.4198707670516, 1332.2611445414686,
    1182.0525565291343, 1069.8889340769158, 979.7825798835856,
    912.148820524711, 855.9434437189825, 807.0984177210509,
    774.0291418189877, 603.0255277285801, 534.2010050275243,
    497.81362380207383, 478.84142759120857, 425.445677365369,
    398.737773012851, 384.1881196704431, 379.44576238881905,
    370.4789639172259, 369.0169989678454, 359.8487333986449,
    360.3388605240409, 356.8702115358788, 356.49732737690283,
    357.75022597425567,
]

GOLDEN_TRAJECTORY = {
    'compliance': GOLDEN_COMPLIANCE,
    'clip_loss': GOLDEN_CLIP_LOSS,
    'clip_loss_raw': GOLDEN_CLIP_LOSS_RAW,
    'loss': GOLDEN_TOTAL,
}

# The reference run stopped here, on the compliance-based convergence test,
# well short of its 200-iteration cap.
GOLDEN_FINAL_STEP = 124

# sha256 of the initial image as committed to Venice at 97f9336, so a silently
# re-exported or re-compressed copy fails loudly rather than shifting the run.
GOLDEN_IMAGE_SHA256 = (
    '750bb809f38c0f5ef6f9af51f228a860a40287b56efde07e27dc061eff157d98')

# Characterization pins for the initial design at the schedule's COARSE grid,
# i.e. the field Venice's `init_weight_with_image` actually installs. Measured
# once from the transform chain; they exist to catch a chain that has been
# reordered or had a logit reintroduced, either of which moves these.
COARSE_IMAGE_SHAPE = (1, 64, 32)
COARSE_IMAGE_MIN = 0.0
COARSE_IMAGE_MAX = 0.9787172675132751
COARSE_IMAGE_MEAN = 0.23396341502666473
COARSE_IMAGE_PIN_TOL = 1e-6

# Trajectory tolerances, set from a measured replay rather than guessed. A full
# reproduction on this machine stopped at step 122 (reference: 124), fired both
# upsamples at the same steps the reference log implies, and gave these relative
# errors over the 24 comparable recorded points:
#
#     term             median    p90     max
#     compliance        0.82%   2.00%   2.62%
#     clip_loss_raw     0.37%   0.93%   1.14%
#     clip_loss         0.56%   2.18%   2.54%
#     total             0.55%   2.07%   2.55%
#
# The per-point ceilings below are roughly three times the worst point observed.
# That headroom absorbs an unluckier augmentation draw -- one measurement cannot
# bound the tail of a Monte-Carlo process -- while staying an order of magnitude
# under what an actually broken reproduction produces. The failures this has to
# catch are not subtle: a logit on the initial image, the image loaded at the
# final resolution instead of the coarse one, or the repository's 1e-2 learning
# rate instead of Venice's 0.2 each move the curve by tens of percent or stop it
# converging at all.
#
# The products get the same ceiling as their factors rather than a wider one:
# `clip_loss` is `clip_loss_raw * compliance * clip_alpha`, so its error is
# roughly the sum of the other two, and in the measured run it came out no worse
# than compliance alone.
TRAJECTORY_RTOL = {
    'compliance': 0.08,
    'clip_loss_raw': 0.04,
    'clip_loss': 0.08,
    'loss': 0.08,
}

# The median is the statistic that actually discriminates, because it is stable
# across draws where the max is not: a run that wandered off and happened to
# cross the reference at a point or two fails here even with every individual
# point inside its own ceiling. Also ~3x the measured medians above.
TRAJECTORY_MEDIAN_RTOL = {
    'compliance': 0.025,
    'clip_loss_raw': 0.012,
    'clip_loss': 0.020,
    'loss': 0.020,
}

# How far the convergence step may move. The stopping rule thresholds a noisy
# per-step compliance delta, so it can fire either side of the reference's 124;
# the measured replay stopped at 122. This admits 105 to 142 and so still fails
# a run that quits early or never converges.
CONVERGENCE_STEP_RTOL = 0.15

# Minimum number of the 25 recorded points a replay must reach to be judged. The
# measured replay covered 24. A replay that stops much earlier is a failure in
# its own right (asserted separately), not an excuse to compare four points.
MIN_COMPARED_POINTS = 20

_FULL_RUN_ENV = 'VENICE_PARITY_FULL'


def _clip_weights_are_cached(*model_names) -> bool:
  """Whether every named CLIP checkpoint is already on disk.

  The smoke run must not reach for the network from a test suite, so it is
  skipped rather than triggering a download of several hundred megabytes.
  """
  try:
    from clip import clip as clip_module  # pylint: disable=g-import-not-at-top
  except ImportError:
    return False
  cache = Path(os.path.expanduser('~/.cache/clip'))
  for name in model_names:
    url = clip_module._MODELS.get(name)  # pylint: disable=protected-access
    if url is None or not (cache / os.path.basename(url)).exists():
      return False
  return True


def _skip_reason_for_clip_run():
  """Return why a CLIP-backed run cannot execute here, or None if it can."""
  for module_name in ('clip', 'kornia'):
    if importlib.util.find_spec(module_name) is None:
      return f'{module_name} is not installed'
  config = golden.GOLDEN
  if not _clip_weights_are_cached(config.clip_model_name,
                                  config.clip_rn_model_name):
    return 'CLIP checkpoints are not cached; refusing to download in a test'
  return None


def _relative_errors(replay_values, reference_values):
  """Per-point |replay - reference| / |reference|."""
  replay = np.asarray(replay_values, dtype=np.float64)
  reference = np.asarray(reference_values, dtype=np.float64)
  return np.abs(replay - reference) / np.abs(reference)


class VeniceGoldenLogTest(absltest.TestCase):
  """The reference trajectory and this repository's loss algebra agree.

  This is the one part of parity that is exact arithmetic rather than a
  statistical comparison: given Venice's own compliance and raw CLIP value,
  `venice_compat_total_loss` has to reproduce Venice's own weighted term and
  total. Nothing here runs a model.
  """

  def test_transcribed_series_are_aligned(self):
    self.assertLen(GOLDEN_STEPS, 25)
    self.assertEqual(GOLDEN_STEPS[-1], GOLDEN_FINAL_STEP)
    for name, series in GOLDEN_TRAJECTORY.items():
      self.assertLen(series, len(GOLDEN_STEPS), msg=name)

  def test_algebra_reproduces_every_logged_point(self):
    algebra = VeniceLossAlgebra(clip_alpha=10.0, compliance_weight=1.0)
    for i, step in enumerate(GOLDEN_STEPS):
      with self.subTest(step=step):
        terms = venice_compat_total_loss(
            torch.tensor(GOLDEN_COMPLIANCE[i], dtype=torch.float64),
            torch.tensor(GOLDEN_CLIP_LOSS_RAW[i], dtype=torch.float64),
            clip_alpha=algebra.clip_alpha,
            compliance_weight=algebra.compliance_weight,
        )
        # The log is printed at full double precision, so this is exact up to
        # decimal round-tripping.
        self.assertAlmostEqual(
            float(terms.clip_weight),
            GOLDEN_COMPLIANCE[i] * 10.0, delta=1e-9)
        self.assertAlmostEqual(
            float(terms.clip_loss) / GOLDEN_CLIP_LOSS[i], 1.0, delta=1e-12)
        self.assertAlmostEqual(
            float(terms.total_loss) / GOLDEN_TOTAL[i], 1.0, delta=1e-12)

  def test_total_really_double_counts_the_raw_term(self):
    """Venice sums every populated loss field, so the raw term lands twice.

    Pinned explicitly because it reads like a bug and an over-helpful cleanup
    would "fix" it, changing the objective the reference run optimized.
    """
    for i, step in enumerate(GOLDEN_STEPS):
      with self.subTest(step=step):
        summed = (GOLDEN_COMPLIANCE[i] + GOLDEN_CLIP_LOSS[i]
                  + GOLDEN_CLIP_LOSS_RAW[i])
        self.assertAlmostEqual(summed / GOLDEN_TOTAL[i], 1.0, delta=1e-12)
        without_raw = GOLDEN_COMPLIANCE[i] + GOLDEN_CLIP_LOSS[i]
        self.assertNotAlmostEqual(
            without_raw / GOLDEN_TOTAL[i], 1.0, delta=1e-12)


class VeniceImageInitTest(absltest.TestCase):
  """The design is seeded from the image the way Venice seeds it."""

  def test_asset_is_the_committed_venice_file(self):
    path = golden.GOLDEN_IMAGE_PATH
    self.assertTrue(path.exists(), f'missing initial image: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    self.assertEqual(digest, GOLDEN_IMAGE_SHA256)

  def test_loader_returns_pixel_space_float32(self):
    from neural_structural_optimization.train.utils import (  # pylint: disable=g-import-not-at-top
        load_venice_initial_image)
    image = load_venice_initial_image(
        golden.GOLDEN_IMAGE_PATH, height=64, width=32, invert_image=True)

    self.assertEqual(tuple(image.shape), COARSE_IMAGE_SHAPE)
    # float32 because `ToTensor` emits float32, which is what made the
    # reference run's design parameter float32. See the module docstring.
    self.assertEqual(image.dtype, torch.float32)
    self.assertAlmostEqual(
        float(image.min()), COARSE_IMAGE_MIN, delta=COARSE_IMAGE_PIN_TOL)
    self.assertAlmostEqual(
        float(image.max()), COARSE_IMAGE_MAX, delta=COARSE_IMAGE_PIN_TOL)
    self.assertAlmostEqual(
        float(image.mean()), COARSE_IMAGE_MEAN, delta=COARSE_IMAGE_PIN_TOL)

  def test_no_logit_is_applied(self):
    """The field stays in [0, 1]; a logit transfer would leave that range.

    An earlier loader in `script/run.py` pushed initial images through
    `torch.logit`, which is right for a model whose parameter is logits and
    wrong for this one -- the physics backend squashes the field itself.
    """
    from neural_structural_optimization.train.utils import (  # pylint: disable=g-import-not-at-top
        load_venice_initial_image)
    image = load_venice_initial_image(
        golden.GOLDEN_IMAGE_PATH, height=64, width=32, invert_image=True)
    self.assertGreaterEqual(float(image.min()), 0.0)
    self.assertLessEqual(float(image.max()), 1.0)
    # The logit of this field is not even finite: it contains exact zeros.
    self.assertEqual(float(image.min()), 0.0)

  def test_inversion_precedes_the_grayscale_reduction(self):
    """Venice inverts and THEN reduces to grayscale, and the order is visible.

    The two steps look like they commute, and they would if the luma weights
    summed to one -- but torchvision's are (0.2989, 0.587, 0.114), which sum to
    0.9999, so swapping them shifts the whole field by 1e-4. That is small, and
    it is also exactly the kind of quiet offset that makes a reproduction
    almost work; the loader keeps Venice's order and this pins it.
    """
    from neural_structural_optimization.train.utils import (  # pylint: disable=g-import-not-at-top
        load_venice_initial_image)
    inverted = load_venice_initial_image(
        golden.GOLDEN_IMAGE_PATH, height=64, width=32, invert_image=True).numpy()
    plain = load_venice_initial_image(
        golden.GOLDEN_IMAGE_PATH, height=64, width=32, invert_image=False).numpy()

    luma_sum = 0.2989 + 0.587 + 0.114
    swapped_order = 1.0 - plain
    np.testing.assert_allclose(
        inverted, swapped_order - (1.0 - luma_sum), rtol=0, atol=1e-6)
    self.assertGreater(np.abs(inverted - swapped_order).max(), 1e-5)

  def test_model_is_seeded_at_the_coarse_resolution(self):
    """The image lands on 32x64, not on the final 128x256.

    This is the detail most likely to be got wrong, and it changes the whole
    trajectory: the reference run's image is resampled to the bottom of the
    resolution schedule and reaches the final grid only through the upsamples.
    """
    model = golden.build_model(clip_loss=None)
    self.assertEqual(tuple(model.shape), COARSE_IMAGE_SHAPE)
    self.assertEqual(tuple(model.full_shape), (1, 256, 128))
    self.assertEqual(tuple(model.z.shape), COARSE_IMAGE_SHAPE)
    self.assertEqual(model.z.dtype, torch.float32)
    self.assertTrue(model.z.requires_grad)
    self.assertAlmostEqual(
        float(model.z.mean()), COARSE_IMAGE_MEAN, delta=COARSE_IMAGE_PIN_TOL)

  def test_seeded_design_survives_an_upsample(self):
    """The schedule carries the image up; dtype and range are preserved."""
    model = golden.build_model(clip_loss=None)
    model.upsample()
    self.assertEqual(tuple(model.z.shape), (1, 128, 64))
    self.assertEqual(model.z.dtype, torch.float32)
    self.assertGreaterEqual(float(model.z.min()), 0.0)
    self.assertLessEqual(float(model.z.max()), 1.0)


class VeniceGoldenConfigTest(absltest.TestCase):
  """`script/venice_golden_250214.py` really encodes the logged configuration."""

  def test_structural_problem_matches_the_log(self):
    config = golden.GOLDEN
    self.assertEqual(config.problem_name, 'multistory_building')
    self.assertEqual((config.width, config.height), (128, 256))
    self.assertEqual(config.density, 0.3)
    self.assertEqual(config.interval, 64)
    self.assertEqual(config.filter_width, 2.0)
    self.assertEqual(config.penal, 3.0)
    self.assertEqual(config.seed, 12)
    self.assertEqual(config.device, 'cpu')
    self.assertEqual(config.prompt, 'skeletons')
    self.assertEqual(config.max_iterations, 200)

  def test_physics_args_match_the_log_at_both_ends_of_the_schedule(self):
    model = golden.build_model(clip_loss=None)
    for stage, (nelx, nely, interval) in enumerate(
        [(32, 64, 16), (64, 128, 32), (128, 256, 64)]):
      with self.subTest(stage=stage):
        self.assertEqual(int(model.args['nelx']), nelx)
        self.assertEqual(int(model.args['nely']), nely)
        self.assertEqual(model.structural_params.interval, interval)
        # penal and the cone-filter radius are held across the schedule, as
        # Venice holds them: it rebuilds every stage through the same
        # `specified_task` defaults.
        self.assertEqual(float(model.args['penal']), 3.0)
        self.assertEqual(float(model.args['filter_width']), 2.0)
        self.assertAlmostEqual(float(model.args['volfrac']), 0.3)
      if model.can_upsample:
        model.upsample()
    self.assertFalse(model.can_upsample)

  def test_loss_algebra_is_enabled_with_the_logged_coefficients(self):
    model = golden.build_model(clip_loss=None)
    self.assertIsNotNone(model.venice_loss_algebra)
    self.assertEqual(model.venice_loss_algebra.clip_alpha, 10.0)
    self.assertEqual(model.venice_loss_algebra.compliance_weight, 1.0)

  def test_clip_preset_for_the_golden_run_is_the_reference_preset(self):
    self.assertEqual(golden.venice_clip_preset(golden.GOLDEN),
                     VENICE_CLIP_PRESET)

  def test_optimizer_carries_venices_learning_rate_and_schedule(self):
    model = golden.build_model(clip_loss=None)
    optimizer = golden.build_optimizer(model)
    # 0.2, not the repository default of 1e-2: Venice builds its Adam with
    # `lr=params['lr']` and rebuilds it the same way on every upsample.
    self.assertEqual(optimizer.lr, 0.2)
    self.assertEqual(optimizer.max_iterations, 200)
    self.assertEqual(optimizer.resize_threshold, 0.5)
    self.assertEqual(optimizer.max_resize_iteration, 50)
    self.assertEqual(optimizer.convergence_threshold, 0.05)
    self.assertEqual(optimizer.compliance_weight, 1.0)
    # The algebra owns the coupling; passing clip_alpha here as well would
    # override it by a second route.
    self.assertIsNone(optimizer.clip_alpha)

  def test_model_resolution_schedule_matches_the_log(self):
    model = golden.build_model(clip_loss=None)
    self.assertEqual(model.resize_num, 2)
    self.assertEqual(model.resize_scale, 2)
    self.assertEqual(model.resizes, 0)


class VeniceGoldenPrecisionTest(absltest.TestCase):
  """float32 storage is faithful to Venice and costs nothing in the solve."""

  def test_objective_on_float32_equals_the_float64_objective(self):
    """The autograd backend promotes, so the stored dtype does not change it.

    Venice hands its float32 field straight to the physics; this repository
    casts to float64 first. Both paths compute the same number, which is why
    the float64 cast in `StructuralLoss` is not itself a divergence.
    """
    model = golden.build_model(clip_loss=None)
    field32 = model.z.detach().numpy()
    self.assertEqual(field32.dtype, np.float32)
    value32 = float(np.asarray(batched_topo_loss(field32, [model.env]))[0])
    value64 = float(np.asarray(
        batched_topo_loss(field32.astype(np.float64), [model.env]))[0])
    self.assertEqual(value32, value64)

  def test_tolerances_stay_above_float32_resolution(self):
    """No tolerance here is tight enough for float32 rounding to bind.

    float32 carries ~1.2e-7 relative resolution. Every trajectory tolerance is
    orders of magnitude above that, so the sampling noise dominates and a
    failure here means the trajectory moved, not that a scalar rounded.
    """
    float32_resolution = float(np.finfo(np.float32).eps)
    for name, rtol in TRAJECTORY_MEDIAN_RTOL.items():
      with self.subTest(term=name):
        self.assertGreater(rtol, 1000 * float32_resolution)


class VeniceSmokeRunTest(absltest.TestCase):
  """A few steps of the real wiring, at a size the default suite can afford.

  The numbers this produces are not comparable to the reference log -- the grid,
  the crop count and the CLIP view are all reduced. What it proves is that the
  golden path still assembles and runs: the preset, the image seeding, the
  resolution schedule, and the term breakdown the parity comparison reads.
  """

  def setUp(self):
    super().setUp()
    reason = _skip_reason_for_clip_run()
    if reason is not None:
      self.skipTest(reason)

  def test_reduced_run_produces_a_consistent_term_breakdown(self):
    config = golden.SMOKE
    ds = golden.run(config)

    steps = int(ds.sizes['step'])
    self.assertGreater(steps, 0)
    self.assertLessEqual(steps, config.max_iterations)
    for name in ('loss', 'compliance', 'clip_loss', 'clip_loss_raw',
                 'clip_weight', 'design'):
      self.assertIn(name, ds)

    compliance = ds['compliance'].values
    clip_loss = ds['clip_loss'].values
    clip_raw = ds['clip_loss_raw'].values
    clip_weight = ds['clip_weight'].values
    total = ds['loss'].values

    self.assertTrue(np.all(np.isfinite(total)))
    self.assertTrue(np.all(compliance > 0))
    self.assertTrue(np.all(clip_raw > 0))

    # The recorded breakdown must satisfy Venice's algebra step for step,
    # including the raw term appearing twice. float32 compliance makes this a
    # single-precision, not a double-precision, identity.
    np.testing.assert_allclose(
        clip_weight, compliance * config.clip_alpha, rtol=1e-6)
    np.testing.assert_allclose(clip_loss, clip_raw * clip_weight, rtol=1e-6)
    np.testing.assert_allclose(
        total, compliance + clip_loss + clip_raw, rtol=1e-6)

    # The design is reported on the final grid whatever stage produced it.
    self.assertEqual(
        (ds.sizes['y'], ds.sizes['x']), (config.height, config.width))


class VeniceFullParityTest(absltest.TestCase):
  """The whole logged loss curve, reproduced. Opt-in; see the module docstring."""

  # The replay costs a quarter of an hour, so it runs once for the class and
  # every test reads the same dataset.
  ds = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if os.environ.get(_FULL_RUN_ENV) != '1' or _skip_reason_for_clip_run():
      return
    cls.ds = golden.run()

  def setUp(self):
    super().setUp()
    if os.environ.get(_FULL_RUN_ENV) != '1':
      self.skipTest(
          f'set {_FULL_RUN_ENV}=1 to run the full 128x256 reproduction '
          '(~15 minutes)')
    reason = _skip_reason_for_clip_run()
    if reason is not None:
      self.skipTest(reason)

  def test_replay_tracks_the_reference_trajectory(self):
    ds = self.ds
    steps = int(ds.sizes['step'])

    self.assertTrue(
        bool(ds.attrs['converged']),
        f'replay ran to its {golden.GOLDEN.max_iterations}-iteration cap '
        'instead of converging; the reference stopped at '
        f'{GOLDEN_FINAL_STEP}')
    self.assertAlmostEqual(
        steps / GOLDEN_FINAL_STEP, 1.0, delta=CONVERGENCE_STEP_RTOL,
        msg=f'replay stopped after {steps} steps, reference after '
            f'{GOLDEN_FINAL_STEP}')

    # Venice's step `s` is this repository's zero-based index `s - 1`.
    indices = [(i, s - 1) for i, s in enumerate(GOLDEN_STEPS) if s - 1 < steps]
    self.assertGreaterEqual(len(indices), MIN_COMPARED_POINTS)

    for name, reference in GOLDEN_TRAJECTORY.items():
      with self.subTest(term=name):
        replay = ds[name].values
        errors = _relative_errors(
            [replay[j] for _, j in indices],
            [reference[i] for i, _ in indices])
        worst = int(np.argmax(errors))
        self.assertLess(
            errors.max(), TRAJECTORY_RTOL[name],
            msg=f'{name} diverged at reference step '
                f'{GOLDEN_STEPS[indices[worst][0]]}: replay '
                f'{replay[indices[worst][1]]!r} vs reference '
                f'{reference[indices[worst][0]]!r}')
        self.assertLess(
            float(np.median(errors)), TRAJECTORY_MEDIAN_RTOL[name],
            msg=f'{name} median relative error over {len(errors)} recorded '
                f'points was {np.median(errors):.4f}')

  def test_replay_ends_where_the_reference_ended(self):
    """The endpoint alone, kept separate so a curve failure reads distinctly."""
    ds = self.ds
    for name, reference in GOLDEN_TRAJECTORY.items():
      with self.subTest(term=name):
        self.assertLess(
            abs(float(ds[name].values[-1]) - reference[-1]) / abs(reference[-1]),
            TRAJECTORY_RTOL[name])


if __name__ == '__main__':
  absltest.main()
