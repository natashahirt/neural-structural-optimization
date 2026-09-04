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

Of those four series only two are independent: `clip_loss` is
`clip_loss_raw * compliance * clip_alpha` and `loss` is the sum of all three,
so both are algebraic functions of compliance and the raw CLIP term. The raw
term turns out to be nearly flat with respect to the design -- 2.24% across
blank, noise, stripe and skeleton fields, narrower than its own 4% tolerance --
which leaves COMPLIANCE doing essentially all of the trajectory's geometric
work.

WHY THE CURVE IS NOT ENOUGH, AND WHAT PINS THE DESIGN
-----------------------------------------------------
Compliance is a scalar functional of the design, and a left-right MIRRORED
skeleton is a different design with almost the same one. Measured against this
reference: mirroring matches compliance to 0.041% and `clip_loss_raw` to
0.002%, i.e. better than the reproduction itself does, so it would pass every
assertion on the curve. A harness that checks only the loss series therefore
does not pin the geometry at all -- it pins a projection of it that mirroring
happens to lie inside.

Two design-side comparisons sit beside the curve, both in
`VeniceFullParityTest`. They do different jobs, and the first one is NOT a
geometry pin:

* `params['volume_actual']`, transcribed from the log. This is Venice's
  `get_volume_ratio`: the fraction of the RAW design parameter above 0.9, a
  strict-inequality count and NOT a mean density. The count is permutation-
  invariant -- a left-right mirror of the same field produces the same
  number -- so this pin constrains fill fraction / saturation, not layout.
  The distinction from a mean density is still the whole content of the pin
  as a *statistic*: the rendered mean is held at `volfrac` by the volume
  constraint, so a mean would restate 0.30 and constrain nothing.
* the rendered final design, compared field-to-field against Venice's own
  output image (Pearson correlation, plus a saved side-by-side PNG). The
  correlation floor is deliberately loose -- a 16x16 coarsening or a heavy
  blur of the reference can still clear it -- so it rejects the obviously
  wrong topologies (X-brace, noise) but is not a substitute for looking at
  the image. The saved visual is first-class; do not chase a tighter Pearson
  floor to close that gap.

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
with a fixed error.

The second contribution is not noise at all. It is the one deliberate modelling
difference, noted under `fix_right_wall` in
`structural.problems.multistory_building`: this repository constrains X on the
right wall and Venice does not, because without it the stiffness matrix is
singular and CHOLMOD refuses the solve outright. On the field a replay actually
starts from -- the seeded Venice image -- that constraint biases compliance by
0.335% at 32x64, 0.365% at 64x128 and 0.380% at 128x256. It is a SYSTEMATIC
bias, in one direction, compounding through all 122 gradient steps, not
sampling noise that averages out. An earlier note here quoted 0.07%, which is
the figure on a uniform density 0.3 field and understates the real bias by
roughly a factor of five; the seeded-field measurements above are the relevant
ones and are the reason the tolerances cannot be tightened to the sampling
floor.

WHAT THE REPRODUCTION ACTUALLY ACHIEVES
---------------------------------------
Better than the tolerances require. A full replay converged at step 122 against
the reference's 124, fired its two upsamples at steps 48 and 70, and tracked
the logged curve to a median relative error under 1% on every term, ending
within 0.6% of the reference's final total. The measured error distribution and
the tolerances derived from it are tabulated at `TRAJECTORY_RTOL`.

The upsample steps are a CHARACTERIZATION pin, not a transcription: the golden
JSON has no resize field, so Venice's own timing is not recorded anywhere and
can only be inferred from kinks in the logged compliance curve. See
`GOLDEN_RESIZE_STEPS`.

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
from PIL import Image

# `_resize_short_side` is private, and imported anyway because it is the one
# implementation of Venice's `transforms.Resize(int)` in this repository,
# already pinned bit-exact against torchvision. Venice runs its display image
# and its CLIP view through the SAME call, so re-deriving it here would be a
# second copy of the same claim, free to drift from the one under test.
from neural_structural_optimization.models.loss_clip import (
    VENICE_CLIP_PRESET,
    _resize_short_side,
)
from neural_structural_optimization.models.model_base import (
    VeniceLossAlgebra,
    venice_compat_total_loss,
)
from neural_structural_optimization.models.utils import batched_topo_loss
from neural_structural_optimization.train.optimizers import VENICE_ITERATION_OFFSET

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

# params['volume_actual'], transcribed from the same log. Venice's
# `get_volume_ratio` (`models.py` line 837): the fraction of the RAW design
# parameter strictly above 0.9. At 128x256 this is 10316 of 32768 elements.
GOLDEN_VOLUME_ACTUAL = 0.3148193359375

# sha256 of the initial image as committed to Venice at 97f9336, so a silently
# re-exported or re-compressed copy fails loudly rather than shifting the run.
GOLDEN_IMAGE_SHA256 = (
    '750bb809f38c0f5ef6f9af51f228a860a40287b56efde07e27dc061eff157d98')

# sha256 of Venice's own final design image, likewise copied into this
# repository rather than read across checkouts. It is the ONLY surviving
# artifact of the reference run's geometry -- the log records scalars only --
# so a re-encoded copy would quietly move the one design pin there is.
GOLDEN_FINAL_IMAGE_SHA256 = (
    'd2264973199422722db657983f349c2a2552566133b2cf4ed42195eeb37a2271')

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
# upsamples at the steps pinned in `GOLDEN_RESIZE_STEPS` -- which the reference
# log does NOT record -- and gave these relative errors over the 24 comparable
# recorded points:
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

# Gradient steps at which the two upsamples fired in the measured replay.
#
# This is a CHARACTERIZATION pin. The golden JSON has no resize field, so
# Venice's own timing was never recorded; the older claim that these are the
# steps "the reference log implies" was wrong, and the log can at best be read
# for kinks in the compliance curve. What the pin does hold is that the
# schedule fires where THIS repository's rules say it should, which is
# checkable arithmetic on both steps:
#
#   step 48 -> iteration 48 + 1 + VENICE_ITERATION_OFFSET = 50, and
#              50 % max_resize_iteration == 0, so the PERIODIC trigger fires.
#   step 70 -> iteration 72, and 72 % 50 == 22, so the periodic trigger cannot
#              have fired; this one is the compliance-DELTA trigger.
#
# Both branches of that `or` therefore run in the reference reproduction, which
# is why `test_training_loop_coverage.py` exercises them separately too.
GOLDEN_RESIZE_STEPS = [48, 70]

# Venice's `params['img_width']`: the short side its display and CLIP views are
# both resized to before anything else happens to them.
DISPLAY_SHORT_SIDE = 512

# How far the replay's filled fraction may sit from the logged 0.3148. The
# measured replay gave 0.31644, a relative error of 0.51%; this is ~4x that.
#
# The ceiling also has to stay well under 4.71%, which is where a mean density
# would land (the volume constraint holds that at volfrac = 0.30 whatever the
# design does). Otherwise the pin would pass on a statistic that says nothing
# about fill fraction, which is the failure this whole constant exists to
# prevent -- see `test_the_volume_pin_rejects_both_plausible_wrong_statistics`.
# This pin is permutation-invariant and does not reject a mirror.
VOLUME_RATIO_RTOL = 0.02

# Agreement required between the replay's final design and Venice's, as a
# Pearson correlation over the 128x256 display field. Measured on this machine,
# against the same reference image:
#
#     replay                              0.741
#     the reference vs its OWN mirror     0.632
#     replay, mirrored left-right         0.544
#     the initial image alone             0.466
#     replay, flipped top-to-bottom       0.130
#
# The floor sits below the replay and above the measured wrong answers, but it
# is still a weak pin: a 16x16 coarsening or a sigma=8 blur of the reference
# can clear 0.60. It is kept as a cheap reject of obviously wrong topologies
# (X-brace, noise), not as the geometric claim. The saved side-by-side visual
# is the comparison that actually pins the design; do not raise this floor to
# close the gap.
DESIGN_CORRELATION_MIN = 0.60

# The two margins that make the floor above mean something, rather than being a
# number a blurry-enough field could clear. Measured: 0.197 against the mirror
# and 0.275 against the seed. Differences of correlations are the robust half
# of this comparison -- a different augmentation draw moves both sides together
# -- so these are the assertions that carry the geometric claim.
DESIGN_MIRROR_MARGIN = 0.10
DESIGN_SEED_MARGIN = 0.10

# Correlation is invariant to affine rescaling, so it cannot see a design of the
# right shape at the wrong density. The display field's mean closes that:
# measured 0.3351 against the reference's 0.3323, 0.8% apart.
DISPLAY_MEAN_RTOL = 0.05

# How closely the reference image's own filled fraction has to reproduce the
# logged `volume_actual`. This is not a claim about the replay -- it is the
# check that this file decodes Venice's image the way Venice wrote it. Reading
# the inversion, orientation or channel convention wrong moves it by tens of
# percent. Measured 0.31806 against the logged 0.31482, 1.03% apart, the
# residual being the 4x resample and JPEG quantization the image went through.
REFERENCE_IMAGE_DECODE_RTOL = 0.03

# The seed image is left-right symmetric to within this, measured 0.0002. That
# is what makes the mirror margin above a statement about the OPTIMIZATION: the
# initial condition carries no left-right information for it to inherit.
SEED_MIRROR_SYMMETRY_TOL = 0.01

# Rows carrying a loaded floor of `multistory_building` hold at least this
# multiple of the median row mass. Measured 2.6x to 2.8x on the four floors at
# `interval` = 64; a vertically flipped reading of the image peaks at 1.6x, so
# this is what fixes the image's orientation rather than assuming it.
FLOOR_ROW_MASS_RATIO = 2.0

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


def _block_mean(field, shape):
  """Average `field` down to `shape` over non-overlapping blocks."""
  height, width = shape
  if field.shape[0] % height or field.shape[1] % width:
    raise ValueError(
        f'{field.shape} does not divide evenly into {shape}.')
  block_y, block_x = field.shape[0] // height, field.shape[1] // width
  return field.reshape(height, block_y, width, block_x).mean(axis=(1, 3))


def _venice_display_field(raw_design, short_side=DISPLAY_SHORT_SIDE):
  """Reproduce the field Venice's saved image shows, on the design grid.

  Venice's `structural_model_image_to_PIL_image` (`models.py` line 831) resizes
  the raw design's shorter edge to `params['img_width']`, THEN clamps to
  [0, 1], then inverts. That order is not incidental: the raw parameter is
  unbounded -- the reference run's spans -11.78 to 13.12 -- so resizing before
  clamping lets neighbouring extremes pull an interpolated pixel across the
  bound, and clamping first would give a different field.

  The result is averaged back down to the design grid. That is what makes this
  comparable to the reference image, which went through the same 4x resample:
  clamping in between means the round trip is NOT the identity, so both sides
  have to make it.

  Args:
    raw_design: the raw design parameter, (height, width).
    short_side: Venice's `img_width`.

  Returns:
    The displayed density on the design grid, in [0, 1].
  """
  field = np.ascontiguousarray(raw_design, dtype=np.float32)
  resized = _resize_short_side(
      torch.as_tensor(field)[None, None], short_side).clamp(0.0, 1.0)
  return _block_mean(resized[0, 0].numpy().astype(np.float64), field.shape)


def _reference_image_field():
  """Decode Venice's final image back to a density, at the image's own size.

  Undoes the one invertible step between the saved JPEG and a density: PIL's
  `ImageOps.invert`, so that material reads high rather than low. Nothing else
  is undone -- JPEG quantization is not invertible, and the clamp lost
  everything outside [0, 1] for good, which is why the replay side clamps too.

  The channel convention is checked rather than chosen: Venice expands one
  grayscale channel to three, so the three have to be identical, and taking any
  one of them is then exact where a luma reduction would round.

  Returns:
    The reference density at 512x1024, in [0, 1].
  """
  pixels = np.asarray(Image.open(golden.GOLDEN_FINAL_IMAGE_PATH).convert('RGB'))
  if not (np.array_equal(pixels[..., 0], pixels[..., 1])
          and np.array_equal(pixels[..., 0], pixels[..., 2])):
    raise ValueError(
        'the reference image is not grayscale, so Venice did not write it the '
        'way this decoding assumes.')
  return 1.0 - pixels[..., 0].astype(np.float64) / 255.0


def _reference_display_field(shape):
  """The decoded reference image averaged back down to the design grid.

  The 4x resample Venice applied on the way out is undone by block-averaging,
  which also suppresses most of the JPEG ringing -- it survives only at the
  edges of a field that is 97% saturated.

  Pixel-level statistics such as the filled fraction must NOT be taken here:
  averaging blurs the two-thirds of a pixel-count that sits at an edge, and it
  moves the fraction above 0.9 from 0.318 to 0.295. Use
  `_reference_image_field` for those.

  Args:
    shape: the design grid, (height, width).

  Returns:
    The reference density on that grid, in [0, 1].
  """
  return _block_mean(_reference_image_field(), shape)


def _correlation(field_a, field_b):
  """Pearson correlation between two fields of the same shape."""
  return float(np.corrcoef(np.ravel(field_a), np.ravel(field_b))[0, 1])


def _seed_display_field(shape):
  """The initial image on the final grid, as a control for the design pin.

  The seeded image is where both runs start, so any agreement it already
  explains is agreement the design comparison has not earned.
  """
  from neural_structural_optimization.train.utils import (  # pylint: disable=g-import-not-at-top
      load_venice_initial_image)
  height, width = shape
  seed = load_venice_initial_image(
      golden.GOLDEN_IMAGE_PATH, height=height, width=width,
      invert_image=golden.GOLDEN.invert_image)
  return _venice_display_field(seed.numpy()[0])


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


class VeniceFinalImageDecodingTest(absltest.TestCase):
  """Venice's final image is read back the way Venice wrote it.

  The design comparison in `VeniceFullParityTest` is only as good as this
  decoding, and a decoding is exactly the kind of thing that is confidently
  wrong: invert the wrong way and the pin measures the VOID, read the rows
  upside down and it measures a different building. Both mistakes produce a
  clean-looking number, so each convention is checked against something the
  reference run recorded independently rather than asserted in a comment.

  None of this needs a replay, CLIP or a physics solve, so it runs in the
  default suite -- which matters, because it is the half of the design pin that
  can be wrong without any replay ever being run.
  """

  DESIGN_SHAPE = (256, 128)

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.image = _reference_image_field()
    cls.reference = _reference_display_field(cls.DESIGN_SHAPE)

  def test_asset_is_the_committed_venice_file(self):
    path = golden.GOLDEN_FINAL_IMAGE_PATH
    self.assertTrue(path.exists(), f'missing reference design image: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    self.assertEqual(digest, GOLDEN_FINAL_IMAGE_SHA256)

  def test_image_geometry_is_the_design_grid_resized_by_the_view_size(self):
    """512x1024, i.e. the 128x256 grid with its short side taken to 512."""
    with Image.open(golden.GOLDEN_FINAL_IMAGE_PATH) as image:
      width, height = image.size
    design_height, design_width = self.DESIGN_SHAPE
    self.assertEqual(width, DISPLAY_SHORT_SIDE)
    self.assertEqual(height, DISPLAY_SHORT_SIDE * design_height // design_width)

  def test_inversion_is_undone_the_right_way_round(self):
    """The filled fraction of the decoded field reproduces the logged one.

    This is the load-bearing convention check. Venice inverts on the way out,
    so material is DARK in the file; undoing that gives a field whose fraction
    above 0.9 should be the logged `volume_actual`. Reading the file without
    undoing the inversion measures the void instead, which is most of the
    picture -- so the two readings are not close, and this test tells them
    apart rather than trusting the sign of a comment.

    Taken at the image's own resolution: a filled fraction is a pixel count,
    and averaging down to the design grid first would fold the blurred edges
    into it and move the answer by 6%.
    """
    filled = float((self.image > 0.9).mean())
    self.assertAlmostEqual(
        filled / GOLDEN_VOLUME_ACTUAL, 1.0, delta=REFERENCE_IMAGE_DECODE_RTOL,
        msg=f'decoded filled fraction {filled} against the logged '
            f'{GOLDEN_VOLUME_ACTUAL}')

    not_inverted = float((1.0 - self.image > 0.9).mean())
    self.assertGreater(
        abs(not_inverted - GOLDEN_VOLUME_ACTUAL) / GOLDEN_VOLUME_ACTUAL,
        10 * REFERENCE_IMAGE_DECODE_RTOL,
        msg='reading the image without undoing the inversion is within '
            'tolerance of the logged volume too, so this test cannot '
            'distinguish the two conventions')

  def test_rows_are_the_right_way_up_against_the_loaded_floors(self):
    """The heavy rows are where `multistory_building` puts its floors.

    `forces[:, ::interval]` loads rows 0, 64, 128 and 192 of a 256-row grid, so
    those rows carry structure. The spacing is deliberately not symmetric about
    the mid-height -- a flipped reading would put the load rows at 255, 191,
    127 and 63 -- so this pins the vertical orientation, which mirroring the
    columns cannot help with and which the volume fraction above is blind to.
    """
    row_mass = self.reference.sum(axis=1)
    median = float(np.median(row_mass))
    floors = list(range(0, self.DESIGN_SHAPE[0], golden.GOLDEN.interval))
    self.assertLen(floors, 4)
    for row in floors:
      with self.subTest(floor_row=row):
        self.assertGreater(row_mass[row], FLOOR_ROW_MASS_RATIO * median)

    flipped = row_mass[::-1]
    self.assertLess(
        min(flipped[row] for row in floors), FLOOR_ROW_MASS_RATIO * median,
        msg='a vertically flipped reading also lands the floors on the heavy '
            'rows, so this test does not constrain the orientation')

  def test_the_seed_image_carries_no_left_right_information(self):
    """Mirroring the initial image does not change how well it matches.

    This is what licenses the mirror margin in `VeniceFullParityTest`. If the
    seeded image were itself lopsided, a replay could inherit the agreement
    without the optimization having produced any of it, and the mirror test
    would be measuring the loader.
    """
    seed = _seed_display_field(self.DESIGN_SHAPE)
    upright = _correlation(seed, self.reference)
    mirrored = _correlation(seed[:, ::-1], self.reference)
    self.assertAlmostEqual(
        upright, mirrored, delta=SEED_MIRROR_SYMMETRY_TOL,
        msg=f'seed correlates {upright} upright and {mirrored} mirrored')

  def test_the_seed_image_does_not_already_explain_the_reference(self):
    """And it leaves room for the design pin to say something.

    A floor the initial condition already clears would be a pin on the loader
    wearing a geometry pin's name.
    """
    seed = _seed_display_field(self.DESIGN_SHAPE)
    self.assertLess(_correlation(seed, self.reference),
                    DESIGN_CORRELATION_MIN - DESIGN_SEED_MARGIN)


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

  def test_clip_preset_fields_match_the_logged_configuration(self):
    """Every preset field, against the LOG rather than against itself.

    This replaces an assertion that `venice_clip_preset(GOLDEN)` equalled
    `VENICE_CLIP_PRESET`, which was worth nothing: three of the four fields are
    unset on both sides, so both were reading the same dataclass defaults and
    the comparison reduced to `x == x`. A preset default edited in
    `loss_clip.py` would have moved both sides together and passed.

    The log carries three of these directly. `min_original_size` and
    `aug_noise` are not logged at all -- they are `GenerateCrops`' defaults
    (`CLIP_utils.py` line 138), which the reference run took as-is
    (`loss_utils.py` line 268, which passes `min_original_size=min(480, 480)`
    and no noise argument) -- so they are transcribed from Venice's source and
    pinned as characterization values.
    """
    preset = golden.venice_clip_preset(golden.GOLDEN)

    # params['use_arcsin_transform'] == true. Venice's other branch averages the
    # untransformed L2 distance, which is a different objective.
    self.assertTrue(preset.use_arcsin_transform)
    # params['num_augs'] == 32.
    self.assertEqual(preset.num_augs, 32)
    # params['img_width'] == 512, i.e. the shorter edge Venice resizes the
    # design to before cropping. For the 128x256 grid that is a 512x1024 view.
    #
    # Transcription is the ONLY thing holding this field. Changing it moves
    # `clip_loss_raw` by a measured 0.019%, against a 4% tolerance on that
    # series -- three orders of magnitude under the bar -- so a wrong view size
    # cannot be detected by the trajectory comparison at all. It still decides
    # which spatial scales the 32 crops sample, and so what the CLIP term asks
    # the design to look like: a gradient-shaping constraint that the loss
    # curve is simply not a witness for.
    self.assertEqual(preset.resize_short_side, 512)

    # Not logged; from Venice's source. See the docstring above.
    self.assertEqual(preset.min_original_size, 480)
    self.assertEqual(preset.aug_noise, 0.1)

  def test_the_reference_preset_is_still_the_golden_configuration(self):
    """`GOLDEN` asks for nothing the module defaults do not already give.

    Kept as a separate, honestly-scoped statement of what the old test was
    reaching for. It says the golden configuration adds no override on top of
    `VENICE_CLIP_PRESET` -- useful, and NOT evidence that either side agrees
    with the log, which the test above is for.
    """
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
                 'clip_weight', 'design', 'final_design_raw'):
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

    # The raw parameter is carried too, but on the stage grid the run ENDED on
    # rather than on the final one -- four steps is not enough to reach either
    # upsample here, so it stays coarse. `design` is repeated up to the final
    # grid and the raw field is not, which is the distinction this asserts.
    raw = ds['final_design_raw'].values
    stages_left = config.resize_num - len(ds.attrs['resize_steps'])
    divisor = config.resize_scale ** stages_left
    self.assertEqual(
        raw.shape, (config.height // divisor, config.width // divisor))
    self.assertTrue(np.all(np.isfinite(raw)))


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

  def test_replay_agrees_at_the_last_step_both_runs_recorded(self):
    """The endpoint comparison, at a MATCHED iteration.

    This is the check the previous version of this file did not have. It
    compared the replay's last point against the reference's last point, which
    are different iterations -- 122 against 124 in the measured replay -- and
    since the curve is still falling there, that comparison charges the
    reproduction for two steps of convergence it never claimed to take. The
    logged 73.99691670938864 was consequently never compared at its own
    iteration by anything in this file.

    Here the comparison happens at the latest recorded step the replay actually
    reached. If a replay runs to 124 or beyond, that step IS 124 and the logged
    final point is compared where it was logged.
    """
    ds = self.ds
    steps = int(ds.sizes['step'])
    matched = [(i, s) for i, s in enumerate(GOLDEN_STEPS) if s - 1 < steps]
    self.assertTrue(matched, 'replay reached none of the recorded steps')
    index, step = matched[-1]

    for name, reference in GOLDEN_TRAJECTORY.items():
      with self.subTest(term=name):
        replay = float(ds[name].values[step - 1])
        self.assertLess(
            abs(replay - reference[index]) / abs(reference[index]),
            TRAJECTORY_RTOL[name],
            msg=f'{name} at reference step {step}: replay {replay!r} vs '
                f'reference {reference[index]!r}')

  def test_both_runs_final_points_agree_despite_stopping_at_different_steps(self):
    """Each run's OWN last point, which is not a matched-iteration comparison.

    Kept, renamed, and scoped honestly. Both runs stop on a noisy compliance
    delta, so they stop at different iterations -- 122 against 124 -- and this
    asks the weaker question of whether they came to rest in the same place.
    `CONVERGENCE_STEP_RTOL` is what bounds how far apart those steps may be;
    the matched comparison above is what bounds the curve.
    """
    ds = self.ds
    steps = int(ds.sizes['step'])
    for name, reference in GOLDEN_TRAJECTORY.items():
      with self.subTest(term=name):
        self.assertLess(
            abs(float(ds[name].values[-1]) - reference[-1]) / abs(reference[-1]),
            TRAJECTORY_RTOL[name],
            msg=f'{name}: replay resting value after {steps} steps against '
                f'the reference resting value after {GOLDEN_FINAL_STEP}')

  def test_both_upsamples_fire_where_the_measured_replay_fires_them(self):
    """The resolution schedule, and which trigger fired each time.

    A characterization pin -- the golden JSON records no resize timing at all.
    What is checkable is the arithmetic of this repository's own rules, so the
    attribution of each step to a trigger is asserted rather than annotated:
    step 48 lands on the period, step 70 provably cannot.
    """
    ds = self.ds
    self.assertEqual(list(ds.attrs['resize_steps']), GOLDEN_RESIZE_STEPS)

    period = golden.GOLDEN.max_resize_iteration
    periodic, delta = GOLDEN_RESIZE_STEPS
    self.assertEqual((periodic + 1 + VENICE_ITERATION_OFFSET) % period, 0)
    self.assertNotEqual((delta + 1 + VENICE_ITERATION_OFFSET) % period, 0)

  def test_replay_reproduces_the_logged_volume_fraction(self):
    """`params['volume_actual']`, on the raw field Venice measured it on.

    Venice's `get_volume_ratio` counts elements strictly above 0.9 in the RAW
    design parameter. `golden.venice_volume_ratio` is that definition; using a
    mean density instead would be a different statistic that happens to sit
    nearby, and `test_the_volume_pin_rejects_both_plausible_wrong_statistics`
    is why the distinction is not cosmetic.

    This is a fill-fraction pin, not a geometry pin: the count is
    permutation-invariant, so a left-right mirror of the same field produces
    the same number. Layout is checked by the image comparison (and the
    saved visual), not here.
    """
    ds = self.ds
    replay = golden.venice_volume_ratio(ds['final_design_raw'].values)
    self.assertAlmostEqual(
        replay / GOLDEN_VOLUME_ACTUAL, 1.0, delta=VOLUME_RATIO_RTOL,
        msg=f'replay filled fraction {replay} against the logged '
            f'{GOLDEN_VOLUME_ACTUAL}')

  def test_the_volume_pin_rejects_both_plausible_wrong_statistics(self):
    """Neither a mean density nor the rendered field reproduces the log.

    Two near-misses are available and both are wrong:

    * the MEAN of the rendered design. `render(volume_constraint=True)` holds
      it at `volfrac` by construction, so a mean-density version of the test
      above would assert the volume constraint against itself and pass for any
      design whatsoever. It also does not match: 0.30 is 4.7% from the logged
      0.3148, which is why `VOLUME_RATIO_RTOL` cannot be widened much further.
    * the filled fraction of the RENDERED design rather than the raw one. That
      is the right statistic on the wrong field -- the constraint has already
      squashed it -- and it reads 0.252, 20% low.

    Measured 0.51% for the correct reading, so the pin discriminates by a
    factor of nine against the nearer of the two.
    """
    ds = self.ds
    rendered = ds['design'].values[-1]

    mean_density = float(np.mean(rendered))
    self.assertAlmostEqual(mean_density, golden.GOLDEN.density, delta=1e-3)
    self.assertGreater(
        abs(mean_density - GOLDEN_VOLUME_ACTUAL) / GOLDEN_VOLUME_ACTUAL,
        VOLUME_RATIO_RTOL)

    rendered_ratio = golden.venice_volume_ratio(rendered)
    self.assertGreater(
        abs(rendered_ratio - GOLDEN_VOLUME_ACTUAL) / GOLDEN_VOLUME_ACTUAL,
        VOLUME_RATIO_RTOL)

  def test_replay_reproduces_the_reference_design_and_not_its_mirror(self):
    """The design itself, against Venice's own final image.

    The one assertion in this file that is not invariant under a left-right
    flip, and the reason it exists: the mirrored skeleton matches compliance to
    0.041% and `clip_loss_raw` to 0.002%, so every other assertion here passes
    on it. Both sides go through Venice's display transform, so the comparison
    is between two fields that have taken the same route.

    Three assertions, in increasing order of how much they carry:

    1. the fields correlate above a floor set below every wrong answer;
    2. they correlate BETTER than the mirrored replay does, by a margin;
    3. and better than the initial image does, by a margin -- otherwise the
       agreement could be inherited from the seed rather than produced.
    """
    ds = self.ds
    reference = _reference_display_field(
        (golden.GOLDEN.height, golden.GOLDEN.width))
    replay = _venice_display_field(ds['final_design_raw'].values)
    self.assertEqual(replay.shape, reference.shape)

    upright = _correlation(replay, reference)
    mirrored = _correlation(replay[:, ::-1], reference)
    seeded = _correlation(
        _seed_display_field(reference.shape), reference)

    self.assertGreater(
        upright, DESIGN_CORRELATION_MIN,
        msg=f'replay design correlates {upright} with the reference')
    self.assertGreater(
        upright - mirrored, DESIGN_MIRROR_MARGIN,
        msg=f'replay design correlates {upright} upright and {mirrored} '
            'mirrored, so this comparison is close to mirror-invariant and '
            'does not pin the geometry')
    self.assertGreater(
        upright - seeded, DESIGN_SEED_MARGIN,
        msg=f'replay design correlates {upright} against {seeded} for the '
            'initial image alone, so the agreement is not evidence the '
            'optimization reproduced anything')

  def test_replay_design_holds_the_reference_density_scale(self):
    """The scale correlation is blind to.

    Pearson correlation is invariant to an affine rescaling of either field, so
    the comparison above would accept the right shape at the wrong density.
    The displayed mean is the missing constraint; unlike the volume fraction it
    counts partly-filled pixels, so the two are not the same statistic.
    """
    ds = self.ds
    reference = _reference_display_field(
        (golden.GOLDEN.height, golden.GOLDEN.width))
    replay = _venice_display_field(ds['final_design_raw'].values)
    self.assertAlmostEqual(
        float(replay.mean()) / float(reference.mean()), 1.0,
        delta=DISPLAY_MEAN_RTOL)


if __name__ == '__main__':
  absltest.main()
