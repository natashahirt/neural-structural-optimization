---
id: chec-1b4b0f34
type: checkpoint
project: semantopology_hardfork
parent_id: deci-ba13459e
title: 'Bounded fix: static CLIP weight refused; volume_actual is fill-fraction'
node_label: 'Bounded fix: static CLIP weight refused; volume_ac'
tags: stage-3,bounded-fix,clip_weight,volume_actual
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/train/optimizers.py@7326e77, neural_structural_optimization/tests/test_venice_guards.py@7326e77,
  neural_structural_optimization/tests/test_venice_parity.py@7326e77
session_id: sess-a469ebb9
created_at: '2026-09-04T18:05:57.271032+00:00'
updated_at: '2026-09-04T18:05:57.271032+00:00'
results: '[{"metric": "pytest_passed", "value": 156, "unit": "1", "split": "neural_structural_optimization/tests",
  "window": "no-VENICE_PARITY_FULL", "criterion": "green including new optimizer-weight
  refusals", "source": "pytest-bounded-fix"}, {"metric": "pytest_skipped", "value":
  8, "unit": "1", "split": "neural_structural_optimization/tests", "window": "no-VENICE_PARITY_FULL",
  "criterion": "full golden replay remains gated", "source": "pytest-bounded-fix"}]'
---
AdaptiveAdam, Adam, and LBFGS now refuse a static CLIP weight under the Venice preset instead of swallowing it. clip_weight / clip_weight_max default to None (unset); the default algebra still uses 1.0. Passing 0.0, 1.0, or 1000.0 under the preset raises the same contradictory-couplings error as get_total_loss; constructing without the argument still runs. A late enable_venice_compat_loss after AdaptiveAdam(clip_weight=1000) is refused at optimize().

volume_actual is documented as a fill-fraction / saturation pin only (permutation-invariant; does not reject a mirror). Geometry is the JPEG comparison plus the saved side-by-side visual. The 0.60 Pearson floor is left as-is: a coarsened or blurred reference can still clear it, and tightening it was explicitly out of scope.

Full suite: 156 passed, 8 skipped (VENICE_PARITY_FULL gated). No full golden replay rerun.
