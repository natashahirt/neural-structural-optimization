---
id: expe-d3f4edac
type: experiment
project: semantopology_hardfork
parent_id: expe-cbfd433d
title: Soft skull prior improves physics and CLIP scores but remains weak in physical
  density
node_label: 'Soft skull prior improves physics and CLIP scores '
tags: ''
status: active
open_threads: 0
success: 'false'
files: script/clip_dream_layout.py@86869f1, neural_structural_optimization/tests/test_clip_motif_scale.py@86869f1,
  script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/summary.json@86869f1,
  script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/comparison.png@86869f1
session_id: sess-bd038090
created_at: '2026-09-08T21:17:06.972975+00:00'
updated_at: '2026-09-08T21:17:06.972975+00:00'
results: '[{"metric": "compliance", "value": 78.41509246826172, "unit": "compliance",
  "split": "128x256 soft-rank skull prior plus global CLIP", "window": "137 optimization
  steps", "criterion": "<=85.291 compliance-only baseline", "source": "script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.39110222458839417, "unit": "loss", "split":
  "128x256 final physical design", "window": "final step", "criterion": "<0.4450201988220215
  compliance-only baseline", "source": "script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/summary.json"},
  {"metric": "mass_on_scaffold", "value": 0.7153436777210068, "unit": "fraction",
  "split": "continuous percentile-rank skull preference", "window": "final step",
  "criterion": "descriptive; no preregistered threshold", "source": "script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/summary.json"},
  {"metric": "support_to_load_connected", "value": 1, "unit": "boolean", "split":
  "128x256 final physical density", "window": "final step", "criterion": "=1", "source":
  "script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/summary.json"},
  {"metric": "floating_mass_fraction", "value": 0.054883810094266745, "unit": "fraction",
  "split": "128x256 final physical density", "window": "final step", "criterion":
  "descriptive; no preregistered threshold", "source": "script/resources/results/clip_dream_layout/human_skull_soft_rank_32x16/summary.json"}]'
---
Reused the recognizable 32x16 continuous human-skull dream, converted its unbounded logits to a continuous percentile-rank preference map, and ran it through the 128x256 Stage 6 occupancy anneal with whole-building CLIP retained. The run is connected and beats the recorded compliance-only baseline on compliance and post-hoc CLIP score. The black/white raw replay contains a clear central skull, but the true physical-density panel remains predominantly a tree-like frame with only faint skull traces. Therefore the soft-prior direction is more promising than in-loop binary projection, but it has not yet made the semantic silhouette load-bearing. The next controlled ablation should remove active CLIP while keeping the same soft initialization/prior, then compare physical-density retention; after that, change the prior objective to weight skull-defining interior/edge regions rather than merely maximizing average continuous preference.
