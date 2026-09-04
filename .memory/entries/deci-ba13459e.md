---
id: deci-ba13459e
type: decision
project: semantopology_hardfork
parent_id: plan-714620c8
title: Parity harness does not yet pin the design; fix cycle scoped
node_label: Parity harness does not yet pin the design; fix cy
tags: stage-3,venice-parity,critic,harness
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/tests/test_venice_parity.py@2b9b438, neural_structural_optimization/train/optimizers.py@2b9b438,
  neural_structural_optimization/models/loss_clip.py@2b9b438, script/venice_golden_250214.py@2b9b438
session_id: sess-2b965d62
created_at: '2026-09-04T13:54:23.692448+00:00'
updated_at: '2026-09-04T13:54:23.692448+00:00'
results: '[{"metric": "replay_final_compliance", "value": 73.95, "unit": "1", "split":
  "venice-golden-250214-full-replay", "window": "128x256-122-steps", "criterion":
  "within harness tolerance of 73.99691670938864", "source": "VeniceFullParityTest"},
  {"metric": "clip_loss_raw_geometry_spread", "value": 2.24, "unit": "%", "split":
  "golden-clip-config-arbitrary-designs", "window": "4-draws-each", "criterion": "must
  be narrower than 4% per-point tolerance to prove the finding", "source": "critic-75c3d63e"},
  {"metric": "mirror_compliance_delta", "value": 0.041, "unit": "%", "split": "seeded-32x64-multistory-right-wall-on",
  "criterion": "<2.5% median / <8% per-point so a mirror would pass", "source": "critic-75c3d63e"},
  {"metric": "fix_right_wall_bias_seeded", "value": 0.38, "unit": "%", "split": "seeded-field-128x256-superlu",
  "window": "vs Venice unconstrained BCs", "criterion": "compare to 0.07% uniform-density
  citation", "source": "critic-75c3d63e"}]'
---
The Stage 3 implementation wave reproduced the 250214 golden *loss curve* (full 128x256 replay: compliance 73.95 vs 73.997, 0.06% error, converged step 122 vs 124). The critic then FAIL'd the claim that this is proven design parity.

Accepted P1s (each with a concrete reproduction):
1. The CLIP-preset test is tautological (both sides read the same dataclass defaults). Flipping use_arcsin_transform / aug_noise / min_original_size leaves the default suite green.
2. clip_loss_raw varies only 2.24% across blank/noise/stripe/skeleton fields, narrower than its own 4% per-point tolerance, so that half of the curve constrains almost nothing.
3. ds['design'] is never compared to a reference. A left-right mirror matches compliance to 0.041% and clip_loss_raw to 0.002%. The log's unused volume_actual=0.3148193359375 (fraction of the raw field above 0.9) is the cheapest unused geometry pin.
4. Recorded replay resize_steps are [48, 70]; step 70 is the untested delta trigger, not the periodic one. The golden JSON has no resize field.
5. AdaptiveAdam_Optimizer.optimize() raises ValueError on a second call (design length 3 vs loss length 6).

P2s accepted as in-scope for the same cycle: fix_right_wall bias on the *seeded* field is 0.335/0.365/0.380% (not the 0.07% measured on uniform density 0.3); the endpoint test compares replay step 122 to reference step 124; clip_weight is silently swallowed under the Venice algebra.

Closed cleanly and must not be reopened: exact log transcription; undetached weight is a real gradient path (4.2e-08 relative vs analytic, 4.80x compliance-gradient amplification); no hardfork CLIP extras leak; both resizes bit-identical to torchvision.

Do not unify the three optimizers' clip_alpha handling: Adam/LBFGS raise because there clip_alpha is inverse+capped; AdaptiveAdam accepts it as the same proportional formula.

The original Stage 3 coordinator run (run-841e6bfd) is gone from process state after overnight idle. Fix cycle continues as a new graph under this phase.
