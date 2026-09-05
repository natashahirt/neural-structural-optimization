---
id: expe-083c1ce7
type: experiment
project: semantopology_hardfork
parent_id: plan-3d516df5
title: Physical-density motif A/B makes skeleton language structural
node_label: Physical-density motif A/B makes skeleton language
tags: clip-scale,physical-density,no-occupancy,structural-geometry,experiment
status: active
open_threads: 0
success: 'true'
files: neural_structural_optimization/models/loss_clip.py@9816779, neural_structural_optimization/models/model_base.py@9816779,
  neural_structural_optimization/tests/test_clip_motif_scale.py@9816779, script/venice_golden_250214.py@9816779,
  script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json@9816779,
  script/resources/results/clip_motif_scale_physical_density_no_occupancy/comparison.png@9816779,
  script/resources/results/clip_motif_scale_physical_density_no_occupancy/physical_density.png@9816779,
  script/resources/results/clip_motif_scale_physical_density_no_occupancy/sketch_run.png@9816779
session_id: sess-dac77cc3
created_at: '2026-09-05T19:08:08.449071+00:00'
updated_at: '2026-09-05T19:08:20.239543+00:00'
results: '[{"metric": "compliance", "value": 72.32072448730469, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "criterion": "at or better than Venice selected-look band",
  "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "mean_physical_density", "value": 0.300104022026062, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "criterion": "hold volfrac 0.30", "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "volume_actual", "value": 0.3114013671875, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.7918543815612793, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "clip_motif_f1p0000", "value": 0.3704879581928253, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "clip_motif_f0p2500", "value": 0.39630183577537537, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "clip_motif_f0p0625", "value": 0.4228254556655884, "split": "venice-250214-motif-scale-physical-density-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_physical_density_no_occupancy/summary.json"},
  {"metric": "test_suite", "value": 240, "unit": "passed", "split": "neural_structural_optimization/tests",
  "window": "2026-09-05", "criterion": "all collected tests pass", "source": "pytest"}]'
related_to: expe-2293aefa
---
Implemented the approved isolated A/B: Venice's primary RandomResizedCrop still receives raw design variables, while only the opt-in physical motif-scale path receives filtered, volume-constrained density. Empty motif scales remain on the frozen raw-only Venice route. Added routing regressions and explicit raw-versus-physical output panels.

No-occupancy run, same seed/scales/crops/weight as expe-2293aefa: converged in 109 steps. Compliance improved from 80.042 to 72.321; volume_actual moved 0.3220 to 0.3114; mean physical density held at 0.3001. Last-step crop losses were building 0.3705, storey 0.3963, member 0.4228 (not numerically comparable to the prior raw-input crop losses because the crop input changed).

Visual result: the final filtered physical-density panel retains a recognizable central skull, rib/spine-like top beam, and tooth/bone texture in the lower horizontal member. Fine anatomy softens under filtering, but motif language now changes the load-bearing density rather than only decorating raw z. This validates the structural-coupling hypothesis. The comparison is in script/resources/results/clip_motif_scale_physical_density_no_occupancy/comparison.png. Full suite: 240 passed, 8 skipped, 241 subtests passed.
