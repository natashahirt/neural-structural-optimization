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

"""Unconditional coverage for `train.optimizers` and the Venice schedule rules.

WHY THIS FILE EXISTS
--------------------
Until the Venice parity work, no test in this suite imported
`neural_structural_optimization/train/optimizers.py` at all. A dangling
top-level import of a nonexistent function therefore survived two consecutive
"the suite is green" reports: nothing ever executed the import that would have
raised, so the module's brokenness was invisible to the only signal being read.

That blind spot was measured rather than assumed. With `train.optimizers` made
unimportable, the three pre-existing test modules -- `test_autograd`,
`test_discretization` and `test_physics` -- still report 75 passed / 92 subtests
passed and exit 0.

`test_venice_parity.py` closed the IMPORT half of the hole, because it loads
`script/venice_golden_250214.py`, which imports the optimizer at module scope;
under the same poisoned import the suite now fails at collection. It did not
close the EXECUTION half. The only test there that enters the training loop,
`VeniceSmokeRunTest`, skips itself when CLIP is absent or its checkpoints are
uncached, and the full replay is gated behind `VENICE_PARITY_FULL`. Simulating a
machine without CLIP -- a fresh clone, or CI -- the suite reports 92 passed /
3 skipped and exit 0 with `AdaptiveAdam_Optimizer.optimize` never entered.

Everything here is therefore deliberately dependency-free, and none of it can
skip:

* The optimizer module is imported at MODULE scope below, so an unimportable
  optimizer is a collection error rather than a quiet skip.
* `LoopExecutionTest` drives the real Adam loop with `clip_loss=None` on grids
  small enough to finish in well under a second, so the loop body, both
  upsamples and the dataset assembly all execute on every run of the suite.
* Where a semantic term is needed to tell compliance apart from the total loss,
  CLIP is stubbed with a plain callable. No checkpoint, no network, no kornia.
* `UndetachedClipWeightGradientTest` checks the gradient identity the whole
  port turns on, term by term, in a fraction of a second. It used to be
  observable only through the opt-in quarter-hour replay in
  `test_venice_parity.py`, i.e. only as a curve that came out roughly right.
* Both upsample TRIGGERS are exercised: `LoopExecutionTest` drives the periodic
  one and `DeltaTriggeredUpsampleTest` drives the compliance-delta one. They
  are separate branches of one `or`, so a run that fires the period can pass
  while the delta branch is broken outright.

The grids are 8x16 -> 16x32 -> 32x64, a quarter-scale copy of the reference
run's 32x64 -> 64x128 -> 128x256. The numbers mean nothing next to the golden
log; `test_venice_parity.py` owns that comparison. What these prove is that the
code path exists, runs, and obeys the rules it claims to obey.
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import ast
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from absl.testing import absltest

# Imported at module scope ON PURPOSE. This is the assertion the original bug
# needed and did not have: if `train.optimizers` cannot be imported, this file
# fails to collect and the suite cannot report green.
from neural_structural_optimization import train
from neural_structural_optimization.train import optimizers as train_optimizers
from neural_structural_optimization.train.optimizers import (
    VENICE_ITERATION_OFFSET,
    AdaptiveAdam_Optimizer,
    Adam_Optimizer,
    LBFGS_Optimizer,
)
from neural_structural_optimization.models.model_ada import INITIAL_PREV_LOSS
from neural_structural_optimization.models.model_base import VeniceLossAlgebra

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SCRIPT = _REPO_ROOT / 'script' / 'venice_golden_250214.py'
_OPTIMIZERS_SOURCE = (
    Path(train_optimizers.__file__).resolve())


def _load_golden_config_module():
  """Import the golden configuration from `script/`, which is not a package."""
  spec = importlib.util.spec_from_file_location(
      'venice_golden_250214_loop_coverage', _GOLDEN_SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


golden = _load_golden_config_module()

# A quarter-scale copy of the reference schedule: `resize_num=2` divides these
# down to an 8x16 start, so the run visits 8x16, 16x32 and 32x64. `interval`
# scales with the grid, as it does in the real config, to keep the floor count
# of `multistory_building` fixed across stages.
SMALL_WIDTH = 32
SMALL_HEIGHT = 64
SMALL_INTERVAL = 16
SMALL_GRIDS = ((8, 16), (16, 32), (32, 64))

# Every optimizer the training package promises to export. Named explicitly so
# that a class deleted or renamed out of `optimizers.py` fails here rather than
# at some caller's import.
EXPECTED_OPTIMIZERS = (
    'Adam_Optimizer',
    'AdaptiveAdam_Optimizer',
    'LBFGS_Optimizer',
    'MMA_Optimizer',
    'OptimalityCriteria_Optimizer',
)

# `max_resize_iteration=2` fires an upsample on every even iteration counter.
# With Venice's counter starting at 1 (see `VENICE_ITERATION_OFFSET`) the
# zero-based steps that trigger are 0 and 2.
LOOP_MAX_RESIZE_ITERATION = 2
LOOP_STEPS = 6
LOOP_EXPECTED_RESIZE_STEPS = [0, 2]

# The same run at `max_resize_iteration=3`. This spacing is what makes the
# off-by-one observable: with the offset the triggers are steps 1 and 4, and
# without it they would be 2 and 5, so the two implementations cannot both pass.
OFFSET_MAX_RESIZE_ITERATION = 3
OFFSET_STEPS = 7
OFFSET_EXPECTED_RESIZE_STEPS = [1, 4]
OFFSET_NAIVE_RESIZE_STEPS = [2, 5]

# The compliance-delta trigger with the periodic one taken out of reach. The
# iteration counter only ever reaches `DELTA_ONLY_STEPS + VENICE_ITERATION_OFFSET
# + 1`, so a period one above that can never divide it -- asserted in the test
# rather than left to arithmetic in a comment. The threshold is above the
# `INITIAL_PREV_LOSS` sentinel so the delta test is crossed from the very first
# step, which makes the fired steps exact instead of trajectory-dependent.
DELTA_ONLY_STEPS = 4
DELTA_ONLY_MAX_RESIZE_ITERATION = DELTA_ONLY_STEPS + 2
DELTA_ONLY_RESIZE_THRESHOLD = INITIAL_PREV_LOSS * 2
DELTA_ONLY_EXPECTED_RESIZE_STEPS = [0, 1]

# A convergence threshold of zero can never be crossed -- `abs(delta) < 0` is
# false for every delta, including zero -- so a run configured with it stops
# only at `max_iterations`. That keeps the step count of these runs exact
# instead of dependent on how fast compliance happens to settle.
NEVER_CONVERGE = 0.0

# Compliance must fall by clearly more than this over the run. The measured
# drop is to ~45% of the starting value; the bar is loose enough not to depend
# on the exact trajectory and tight enough that a loop which silently fails to
# apply its gradients cannot pass it.
MIN_COMPLIANCE_IMPROVEMENT = 0.75

# The recorded breakdown is assembled from float32 compliance, so the algebra
# is a single-precision identity, not a double-precision one.
ALGEBRA_RTOL = 1e-6

STUB_CLIP_VALUE = 5.0

# An alpha deliberately unequal to the reference run's 10, so an override that
# is silently dropped cannot coincide with the algebra's own value.
OVERRIDE_CLIP_ALPHA = 3.0

# Tolerance on the product-rule gradient identity below, relative to the
# largest component of the gradient. Everything in it is float32 -- the design
# parameter is, and `StructuralLoss` returns the compliance scalar in the input
# dtype -- so the floor is float32 rounding accumulated over three separate
# solves, not anything about the algebra. Measured here at 1.1e-07, which is
# float32's own resolution of 1.2e-07; the critique reported 4.2e-08 on its own
# design, and the difference between those two is rounding, not disagreement.
# This leaves ~90x headroom and still sits four orders of magnitude below the
# discrepancy a DETACHED weight produces.
GRADIENT_IDENTITY_RTOL = 1e-5

# How much bigger the detached-weight gradient's error has to be before the
# identity above counts as evidence of anything. The two differ by exactly
# `clip_alpha * clip_loss_raw * grad(compliance)`, measured here at 38% of the
# whole gradient -- a different search direction, not a rounding difference.
DETACHED_GRADIENT_MIN_ERROR = 0.1


def _small_config(**overrides):
  """The golden configuration shrunk to something a unit test can afford."""
  return golden.VeniceGoldenConfig(
      width=SMALL_WIDTH,
      height=SMALL_HEIGHT,
      interval=SMALL_INTERVAL,
      **overrides)


def _stub_clip_loss(logits):
  """Stand in for a CLIP loss with a constant, so no checkpoint is needed.

  Returned on the logits' own device and dtype so it composes with the real
  algebra. A constant carries no gradient of its own, which is fine: the total
  still requires grad through compliance, and through `clip_weight`, which is
  undetached compliance.
  """
  return logits.new_tensor(STUB_CLIP_VALUE)


def _smooth_stub_clip_loss(logits):
  """Stand in for a CLIP loss with something that has a real gradient.

  `_stub_clip_loss` is constant, so it contributes no gradient of its own and
  cannot show what the raw term does to the total. This one is smooth in the
  design and its gradient is non-uniform, so it cannot coincidentally line up
  with the compliance gradient. The scale is chosen to land in the same range
  as the reference run's `clip_loss_raw`, which runs from 0.43 down to 0.38.
  """
  return STUB_CLIP_VALUE * (logits ** 2).mean()


def _gradient_of(model, loss_fn):
  """Return one scalar loss and its gradient with respect to the design.

  The design parameter is replaced wholesale by `upsample`, so the gradient is
  copied out rather than aliased, and the buffer is cleared first so repeated
  calls on one model cannot accumulate into each other.
  """
  model.zero_grad(set_to_none=True)
  value = loss_fn(model())
  value.backward()
  return float(value.detach()), model.z.grad.detach().clone().numpy()


def _resolve_import_from(node: ast.ImportFrom, package: str) -> str:
  """Resolve an `ImportFrom` node's module, relative or absolute."""
  if not node.level:
    return node.module
  base = package.rsplit('.', node.level - 1)[0] if node.level > 1 else package
  return f'{base}.{node.module}' if node.module else base


class OptimizerModuleImportTest(absltest.TestCase):
  """The optimizer module imports, and everything it names really exists.

  This is the regression guard for the original bug. It is cheap, it touches no
  physics, and it runs unconditionally.
  """

  def test_module_and_package_expose_every_optimizer(self):
    for name in EXPECTED_OPTIMIZERS:
      with self.subTest(optimizer=name):
        self.assertTrue(
            hasattr(train_optimizers, name),
            f'{name} is missing from train.optimizers')
        self.assertTrue(
            hasattr(train, name),
            f'{name} is not re-exported by the train package')
        self.assertIn(name, train.__all__)
        self.assertIs(getattr(train, name), getattr(train_optimizers, name))

  def test_every_name_in_the_package_all_resolves(self):
    """`train.__all__` promises these; a stale entry is a broken import waiting."""
    for name in train.__all__:
      with self.subTest(name=name):
        self.assertTrue(hasattr(train, name), f'{name} in __all__ but absent')

  def test_every_top_level_import_target_resolves(self):
    """No top-level import in `optimizers.py` names something nonexistent.

    Importing the module already proves this for the module as a whole, so this
    exists for the failure it localizes: it reports WHICH name is dangling and
    in which module, which is the part a bare ImportError makes you go and find.

    Only plain top-level statements are examined. An import deliberately
    guarded by `try`/`except ImportError` is an optional dependency and is
    outside the bug class this file is about.
    """
    package = train_optimizers.__package__
    tree = ast.parse(_OPTIMIZERS_SOURCE.read_text())
    checked = 0
    for node in tree.body:
      if isinstance(node, ast.Import):
        for alias in node.names:
          with self.subTest(module=alias.name):
            importlib.import_module(alias.name)
            checked += 1
      elif isinstance(node, ast.ImportFrom):
        source_name = _resolve_import_from(node, package)
        source = importlib.import_module(source_name)
        for alias in node.names:
          with self.subTest(module=source_name, name=alias.name):
            self.assertTrue(
                hasattr(source, alias.name),
                f'{source_name} has no attribute {alias.name!r}, but '
                f'optimizers.py imports it at module scope')
            checked += 1
    # A parse that silently matched nothing would make this test vacuous.
    self.assertGreater(checked, 10)


class LoopExecutionTest(absltest.TestCase):
  """The adaptive training loop actually runs, with no CLIP and no skip.

  One CLIP-free replay is shared by the whole class. It covers the loop body,
  both upsamples, the per-stage rebuild, the term breakdown and the
  multi-resolution dataset assembly -- the code that was previously executed
  only when a CLIP checkpoint happened to be cached.
  """

  ds = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = _small_config(
        max_resize_iteration=LOOP_MAX_RESIZE_ITERATION,
        max_iterations=LOOP_STEPS,
        convergence_threshold=NEVER_CONVERGE)
    cls.ds = golden.run(cls.config, with_clip=False)

  def test_run_records_every_field_the_reference_log_carries(self):
    ds = self.ds
    self.assertEqual(int(ds.sizes['step']), LOOP_STEPS)
    for name in ('loss', 'compliance', 'clip_loss', 'clip_loss_raw',
                 'clip_weight', 'design'):
      with self.subTest(field=name):
        self.assertIn(name, ds)
    self.assertTrue(np.all(np.isfinite(ds['loss'].values)))
    self.assertTrue(np.all(ds['compliance'].values > 0))

  def test_the_loop_actually_reduces_compliance(self):
    """Proves gradients reach the parameter, not merely that the loop ran.

    A loop that computed a loss and never applied it -- a missing `backward`,
    an optimizer rebuilt over the wrong parameters after an upsample -- would
    satisfy every structural assertion in this class but not this one.
    """
    compliance = self.ds['compliance'].values
    self.assertLess(compliance[-1], MIN_COMPLIANCE_IMPROVEMENT * compliance[0])

  def test_both_upsamples_fire_where_the_schedule_predicts(self):
    ds = self.ds
    self.assertEqual(list(ds.attrs['resize_steps']), LOOP_EXPECTED_RESIZE_STEPS)
    # The run stopped on `max_iterations`, not on the convergence test, so the
    # step count above is the schedule's and not an accident of the trajectory.
    self.assertFalse(bool(ds.attrs['converged']))

  def test_design_is_reported_on_the_final_grid(self):
    """Frames recorded at three resolutions still share one set of coordinates."""
    ds = self.ds
    self.assertEqual((ds.sizes['y'], ds.sizes['x']),
                     (SMALL_HEIGHT, SMALL_WIDTH))
    designs = ds['design'].values
    self.assertEqual(designs.shape, (LOOP_STEPS, SMALL_HEIGHT, SMALL_WIDTH))
    self.assertTrue(np.all(np.isfinite(designs)))
    # `render(volume_constraint=True)` bounds the reported density.
    self.assertGreaterEqual(float(designs.min()), 0.0)
    self.assertLessEqual(float(designs.max()), 1.0)

  def test_the_raw_design_is_recorded_and_is_not_the_rendered_one(self):
    """`final_design_raw` is the parameter, before the volume constraint.

    The design pins in `test_venice_parity.py` read this field, because
    Venice's `volume_actual` and its saved image are both functions of the raw
    parameter rather than of the rendered density. That only means something if
    the two are actually different objects, which is what this asserts -- and
    unconditionally, since those pins live behind the opt-in replay.

    Both upsamples fire in this run, so the raw parameter has reached the final
    grid and the two fields are directly comparable.
    """
    ds = self.ds
    raw = ds['final_design_raw'].values
    self.assertEqual(raw.shape, (SMALL_HEIGHT, SMALL_WIDTH))
    rendered = ds['design'].values[-1]
    self.assertGreater(
        float(np.abs(raw - rendered).max()), 0.0,
        'the raw parameter equals the rendered design, so the volume '
        'constraint is not being applied on the way to `design`')
    # The constraint pins the rendered mean at volfrac whatever the design
    # does; nothing pins the raw field's.
    self.assertAlmostEqual(
        float(rendered.mean()), self.config.density, delta=1e-3)

  def test_breakdown_satisfies_the_venice_algebra_step_for_step(self):
    ds = self.ds
    compliance = ds['compliance'].values
    np.testing.assert_allclose(
        ds['clip_weight'].values, compliance * self.config.clip_alpha,
        rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(
        ds['clip_loss'].values,
        ds['clip_loss_raw'].values * ds['clip_weight'].values,
        rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(
        ds['loss'].values,
        compliance + ds['clip_loss'].values + ds['clip_loss_raw'].values,
        rtol=ALGEBRA_RTOL)

  def test_without_clip_the_semantic_terms_degenerate_to_zero(self):
    """The edge case of the algebra: no CLIP means total == compliance.

    The weight is still computed and still equals `compliance * clip_alpha`,
    because it is a property of compliance rather than of the CLIP term. Only
    the terms it multiplies vanish.
    """
    ds = self.ds
    np.testing.assert_array_equal(
        ds['clip_loss_raw'].values, np.zeros(LOOP_STEPS))
    np.testing.assert_array_equal(ds['clip_loss'].values, np.zeros(LOOP_STEPS))
    np.testing.assert_allclose(
        ds['loss'].values, ds['compliance'].values, rtol=ALGEBRA_RTOL)
    self.assertTrue(np.all(ds['clip_weight'].values > 0))


class VeniceIterationOffsetTest(absltest.TestCase):
  """The first upsample lands one step earlier than `max_resize_iteration` reads.

  Venice initializes its iteration counter to 1 and increments before testing,
  so after N gradient steps the counter reads N + 1. That off-by-one is
  reproduced deliberately, which means it needs a test that a "corrected"
  zero-based counter would fail.
  """

  def test_offset_constant_is_the_legacy_one(self):
    self.assertEqual(VENICE_ITERATION_OFFSET, 1)

  def test_upsamples_land_one_step_before_the_naive_schedule(self):
    ds = golden.run(
        _small_config(max_resize_iteration=OFFSET_MAX_RESIZE_ITERATION,
                      max_iterations=OFFSET_STEPS,
                      convergence_threshold=NEVER_CONVERGE),
        with_clip=False)
    resize_steps = list(ds.attrs['resize_steps'])
    self.assertEqual(resize_steps, OFFSET_EXPECTED_RESIZE_STEPS)
    self.assertNotEqual(resize_steps, OFFSET_NAIVE_RESIZE_STEPS)


class DeltaTriggeredUpsampleTest(absltest.TestCase):
  """The compliance-delta upsample trigger fires, on its own.

  `AdaptiveAdam_Optimizer` advances a stage when the iteration count divides
  `max_resize_iteration` OR compliance moved less than `resize_threshold`.
  Those are two branches of one `or`, and every other loop-level test in this
  file drives the periodic branch -- so the delta branch could be inverted,
  reading the total loss, or dropped altogether and the suite would stay green.
  The reference run needs both: its first upsample is periodic (step 48, at
  iteration 50) and its second is not (step 70, at iteration 72).

  Here the period is put out of reach and the threshold above the `prev_loss`
  sentinel, so the delta branch is the only thing that can fire and the steps
  it fires on are exact.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = _small_config(
        max_resize_iteration=DELTA_ONLY_MAX_RESIZE_ITERATION,
        max_iterations=DELTA_ONLY_STEPS,
        resize_threshold=DELTA_ONLY_RESIZE_THRESHOLD,
        convergence_threshold=NEVER_CONVERGE)
    cls.ds = golden.run(cls.config, with_clip=False)

  def test_the_periodic_trigger_cannot_account_for_these_upsamples(self):
    """No iteration this run reaches divides the period, so the delta did it.

    Asserted rather than argued in a comment: it is the whole basis for
    attributing the upsamples below to the delta branch.
    """
    reached = [step + 1 + VENICE_ITERATION_OFFSET
               for step in range(DELTA_ONLY_STEPS)]
    for iteration in reached:
      with self.subTest(iteration=iteration):
        self.assertNotEqual(iteration % DELTA_ONLY_MAX_RESIZE_ITERATION, 0)

  def test_both_upsamples_fire_on_the_compliance_delta(self):
    self.assertEqual(list(self.ds.attrs['resize_steps']),
                     DELTA_ONLY_EXPECTED_RESIZE_STEPS)
    # The run went the distance, so the schedule is what stopped advancing --
    # it ran out of upsamples -- rather than the loop ending early.
    self.assertEqual(int(self.ds.sizes['step']), DELTA_ONLY_STEPS)
    self.assertFalse(bool(self.ds.attrs['converged']))

  def test_the_design_still_reaches_the_final_grid(self):
    """A delta-driven schedule visits the same resolutions as a periodic one."""
    self.assertEqual((self.ds.sizes['y'], self.ds.sizes['x']),
                     (SMALL_HEIGHT, SMALL_WIDTH))
    self.assertEqual(self.ds['final_design_raw'].shape,
                     (SMALL_HEIGHT, SMALL_WIDTH))


class UndetachedClipWeightGradientTest(absltest.TestCase):
  """The CLIP weight is undetached, checked on the gradient itself.

  This is the load-bearing dynamic of the whole port. With
  `C = compliance`, `R = clip_loss_raw` and `alpha = clip_alpha`, the legacy
  total is `C + alpha*C*R + R`, so the product rule gives::

      grad(total) = (1 + alpha*R) * grad(C) + (alpha*C + 1) * grad(R)

  Detaching the weight -- the obvious "fix", since a loss weight that carries
  gradient reads like a mistake -- drops the `alpha*R * grad(C)` term and
  leaves `grad(C) + (alpha*C + 1) * grad(R)`. At the reference run's
  coefficients that is a factor of `1 + 10*0.38`, i.e. nearly five times too
  small in the compliance direction, which is a different optimization problem
  rather than a slightly different one.

  Until now this was protected only by the opt-in quarter-hour replay in
  `test_venice_parity.py`, and then only indirectly, through whether the loss
  curve came out close enough. Here it is arithmetic, and it takes three FEA
  solves on an 8x16 grid.
  """

  def setUp(self):
    super().setUp()
    self.config = _small_config()
    self.model = golden.build_model(clip_loss=None, config=self.config)
    self.model.clip_loss = _smooth_stub_clip_loss
    self.alpha = self.model.venice_loss_algebra.clip_alpha

    self.compliance, self.grad_compliance = _gradient_of(
        self.model, self.model.get_structural_loss)
    self.clip_raw, self.grad_clip_raw = _gradient_of(
        self.model, self.model.get_semantic_loss)
    _, self.grad_total = _gradient_of(
        self.model,
        lambda logits: self.model.get_venice_compat_losses(logits).total_loss)

  def _relative_error(self, candidate):
    """Error against the measured total gradient, on the gradient's own scale."""
    return (float(np.abs(self.grad_total - candidate).max())
            / float(np.abs(self.grad_total).max()))

  def test_the_stub_is_not_degenerate(self):
    """Both gradients have to be present for the identity to say anything.

    A constant semantic term (gradient zero) or a compliance-free design would
    make the comparison below pass against formulas it should reject.
    """
    self.assertGreater(np.abs(self.grad_compliance).max(), 0.0)
    self.assertGreater(np.abs(self.grad_clip_raw).max(), 0.0)
    self.assertGreater(self.compliance, 0.0)
    self.assertGreater(self.clip_raw, 0.0)
    # The undetached term this test exists for is alpha*R*grad(C); it has to be
    # a real part of the gradient, not a perturbation of it.
    self.assertGreater(self.alpha * self.clip_raw, 1.0)

  def test_total_gradient_obeys_the_product_rule_over_both_terms(self):
    expected = ((1.0 + self.alpha * self.clip_raw) * self.grad_compliance
                + (self.alpha * self.compliance + 1.0) * self.grad_clip_raw)
    self.assertLess(self._relative_error(expected), GRADIENT_IDENTITY_RTOL)

  def test_a_detached_weight_would_be_rejected(self):
    """Without this the test above could be passing on a coincidence.

    The detached form is the plausible wrong answer -- a loss weight carrying
    gradient reads like a bug -- so it is checked to be plainly outside the
    tolerance rather than merely left unasserted. Measured 38% against a
    tolerance of 0.001%.
    """
    detached = (self.grad_compliance
                + (self.alpha * self.compliance + 1.0) * self.grad_clip_raw)
    self.assertGreater(self._relative_error(detached),
                       DETACHED_GRADIENT_MIN_ERROR)


class ScheduleRulesReadComplianceTest(absltest.TestCase):
  """Both schedule rules threshold COMPLIANCE, never the total loss.

  A CLIP-free run cannot show this, because there the total loss IS compliance.
  A stub semantic term makes them differ by more than an order of magnitude, so
  a rule reading the wrong series is unmissable.
  """

  def test_threshold_tests_receive_compliance_not_the_total(self):
    config = _small_config(max_iterations=3, convergence_threshold=NEVER_CONVERGE)
    golden.seed_everything(config.seed)
    model = golden.build_model(clip_loss=None, config=config)
    model.clip_loss = _stub_clip_loss

    observed = []
    real_threshold_crossed = model.threshold_crossed

    def recording_threshold_crossed(compliance, threshold):
      observed.append(compliance)
      return real_threshold_crossed(compliance, threshold)

    model.threshold_crossed = recording_threshold_crossed

    golden.seed_everything(config.seed)
    ds = golden.build_optimizer(model, config).optimize()

    compliance = list(ds['compliance'].values)
    total = list(ds['loss'].values)
    self.assertEqual(observed, compliance)
    # Guard against a degenerate stub: if the two series were close, agreeing
    # with compliance would not be evidence of anything.
    self.assertGreater(min(total), 10 * max(compliance))
    for seen in observed:
      self.assertNotIn(seen, total)

  def test_stub_keeps_the_algebra_intact(self):
    """The recorded breakdown still obeys the algebra with a semantic term."""
    config = _small_config(max_iterations=2, convergence_threshold=NEVER_CONVERGE)
    golden.seed_everything(config.seed)
    model = golden.build_model(clip_loss=None, config=config)
    model.clip_loss = _stub_clip_loss
    golden.seed_everything(config.seed)
    ds = golden.build_optimizer(model, config).optimize()

    compliance = ds['compliance'].values
    np.testing.assert_allclose(
        ds['clip_loss_raw'].values, np.full(compliance.shape, STUB_CLIP_VALUE),
        rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(
        ds['loss'].values,
        compliance * (1.0 + STUB_CLIP_VALUE * config.clip_alpha)
        + STUB_CLIP_VALUE,
        rtol=ALGEBRA_RTOL)

  def test_threshold_crossed_is_a_delta_against_prev_loss(self):
    model = golden.build_model(clip_loss=None, config=_small_config())
    model.prev_loss = 100.0
    self.assertTrue(model.threshold_crossed(100.4, 0.5))
    self.assertFalse(model.threshold_crossed(99.4, 0.5))
    # Symmetric: an increase of the same size counts as movement too.
    self.assertFalse(model.threshold_crossed(100.6, 0.5))
    # A zero threshold is never crossed, not even by a zero delta.
    self.assertFalse(model.threshold_crossed(100.0, 0.0))

  def test_initial_prev_loss_holds_both_tests_closed_on_the_first_step(self):
    """The sentinel is larger than any first-step compliance, by design."""
    model = golden.build_model(clip_loss=None, config=_small_config())
    self.assertEqual(model.prev_loss, INITIAL_PREV_LOSS)
    first_step_compliance = float(
        model.get_structural_loss(model()).detach())
    self.assertLess(first_step_compliance, INITIAL_PREV_LOSS)
    self.assertFalse(model.threshold_crossed(first_step_compliance, 0.5))
    self.assertFalse(model.threshold_crossed(first_step_compliance, 0.05))

  def test_convergence_stops_the_run_once_the_schedule_is_exhausted(self):
    """With no upsamples left and an unmissable threshold, the run stops at once."""
    config = _small_config(
        resize_num=0, max_iterations=LOOP_STEPS,
        convergence_threshold=INITIAL_PREV_LOSS * 2)
    ds = golden.run(config, with_clip=False)
    self.assertTrue(bool(ds.attrs['converged']))
    self.assertEqual(int(ds.sizes['step']), 1)
    self.assertEqual(list(ds.attrs['resize_steps']), [])


class ScheduleGeometryTest(absltest.TestCase):
  """The resolution schedule visits the grids it says it will."""

  def test_every_stage_rebuilds_the_problem_at_its_own_grid(self):
    model = golden.build_model(clip_loss=None, config=_small_config())
    for stage, (nelx, nely) in enumerate(SMALL_GRIDS):
      with self.subTest(stage=stage):
        self.assertEqual(int(model.args['nelx']), nelx)
        self.assertEqual(int(model.args['nely']), nely)
        self.assertEqual(tuple(model.z.shape), (1, nely, nelx))
        # float32 throughout, as the reference run's parameter was.
        self.assertEqual(model.z.dtype, torch.float32)
      if model.can_upsample:
        model.upsample()
    self.assertFalse(model.can_upsample)
    self.assertEqual(tuple(model.full_shape), (1, SMALL_HEIGHT, SMALL_WIDTH))

  def test_upsample_refuses_once_the_schedule_is_exhausted(self):
    model = golden.build_model(clip_loss=None, config=_small_config())
    while model.can_upsample:
      model.upsample()
    with self.assertRaisesRegex(RuntimeError, 'upsample called after'):
      model.upsample()


class ClipAlphaUnderThePresetTest(absltest.TestCase):
  """`clip_alpha` means two different things, and only one of them conflicts.

  On `Adam_Optimizer` and `LBFGS_Optimizer` it scales a weight INVERSELY
  proportional to compliance and capped at `CLIP_DYNAMIC_WEIGHT_MAX`. That is a
  structurally different formula from the preset's -- proportional, undetached,
  uncapped -- so setting both describes two different runs and is refused.

  On `AdaptiveAdam_Optimizer` it is the SAME formula the preset uses, so it is
  documented to override the algebra's own alpha rather than conflict with it.
  The asymmetry is deliberate; both halves are pinned here because the obvious
  "clean up" is to make the three optimizers agree, which would either break
  the override or silently reinterpret a capped inverse weight as a
  proportional one.
  """

  def _model_with_preset(self):
    model = golden.build_model(clip_loss=None, config=_small_config())
    self.assertIsNotNone(model.venice_loss_algebra)
    return model

  def test_conflicting_optimizers_reject_clip_alpha_under_the_preset(self):
    model = self._model_with_preset()
    for factory, name in ((Adam_Optimizer, 'Adam_Optimizer'),
                          (LBFGS_Optimizer, 'LBFGS_Optimizer')):
      with self.subTest(optimizer=name):
        with self.assertRaisesRegex(ValueError, 'contradictory couplings'):
          factory(model, max_iterations=1, clip_alpha=1.0)

  def test_optimize_rejects_a_preset_enabled_after_construction(self):
    """The seam can be flipped on a built model, so the check runs twice.

    Constructing the optimizer against a model with the default algebra is
    legitimate; enabling the preset afterwards makes the pairing contradictory,
    and `optimize` has to notice before it takes a single step.
    """
    model = golden.build_model(clip_loss=None, config=_small_config())
    model.enable_venice_compat_loss(False)
    optimizer = Adam_Optimizer(model, max_iterations=1, clip_alpha=1.0)
    model.enable_venice_compat_loss(VeniceLossAlgebra())
    with self.assertRaisesRegex(ValueError, 'contradictory couplings'):
      optimizer.optimize()

  def test_the_default_coupling_alone_is_accepted(self):
    model = golden.build_model(clip_loss=None, config=_small_config())
    model.enable_venice_compat_loss(False)
    self.assertEqual(
        Adam_Optimizer(model, max_iterations=1, clip_alpha=1.0).clip_alpha, 1.0)

  def test_the_preset_alone_is_accepted(self):
    model = self._model_with_preset()
    self.assertIsNone(Adam_Optimizer(model, max_iterations=1).clip_alpha)
    self.assertIsNone(LBFGS_Optimizer(model, max_iterations=1).clip_alpha)
    self.assertIsNone(AdaptiveAdam_Optimizer(model, max_iterations=1).clip_alpha)

  def test_adaptive_clip_alpha_overrides_the_algebras_own_alpha(self):
    """The documented override, checked on the weight it is supposed to move.

    The model's algebra carries the reference alpha of 10; the optimizer is
    given a different one. The recorded weight must follow the optimizer, and
    must stay `compliance * alpha` -- proportional and uncapped -- rather than
    turning into the capped inverse weight of the other two optimizers.
    """
    config = _small_config(max_iterations=2, convergence_threshold=NEVER_CONVERGE)
    model = self._model_with_preset()
    self.assertEqual(model.venice_loss_algebra.clip_alpha,
                     golden.GOLDEN.clip_alpha)
    self.assertNotEqual(OVERRIDE_CLIP_ALPHA, golden.GOLDEN.clip_alpha)
    model.clip_loss = _stub_clip_loss

    golden.seed_everything(config.seed)
    ds = AdaptiveAdam_Optimizer(
        model,
        max_iterations=config.max_iterations,
        lr=config.lr,
        clip_alpha=OVERRIDE_CLIP_ALPHA,
        compliance_weight=config.compliance_weight,
        resize_threshold=config.resize_threshold,
        max_resize_iteration=config.max_resize_iteration,
        convergence_threshold=config.convergence_threshold).optimize()

    compliance = ds['compliance'].values
    np.testing.assert_allclose(
        ds['clip_weight'].values, compliance * OVERRIDE_CLIP_ALPHA,
        rtol=ALGEBRA_RTOL)
    # And emphatically not the algebra's 10, which is what would happen if the
    # override were dropped on the way through `get_venice_compat_losses`.
    self.assertGreater(
        np.abs(ds['clip_weight'].values
               - compliance * golden.GOLDEN.clip_alpha).min(),
        1.0)


if __name__ == '__main__':
  absltest.main()
