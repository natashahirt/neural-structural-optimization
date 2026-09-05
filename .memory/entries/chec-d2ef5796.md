---
id: chec-d2ef5796
type: checkpoint
project: semantopology_hardfork
parent_id: plan-3d516df5
title: CLIP motif-scale second path landed; Venice baseline frozen
node_label: CLIP motif-scale second path landed; Venice baseli
tags: clip-scale,checkpoint,not-silhouette
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/models/loss_clip.py@9ab1887, neural_structural_optimization/experiment.py@9ab1887,
  script/venice_golden_250214.py@9ab1887, neural_structural_optimization/train/optimizers.py@9ab1887,
  neural_structural_optimization/tests/test_clip_motif_scale.py@9ab1887
session_id: sess-60a28823
created_at: '2026-09-05T06:18:43.888352+00:00'
updated_at: '2026-09-05T06:18:43.888352+00:00'
---
Second CLIP path is live and opt-in. Venice RandomResizedCrop stays the 250214 baseline (empty motif_scale_fracs). physical_motif_scale_fracs(256, 64)=(1.0, 0.25, 0.0625) maps building/storey/member to elevation-height fractions; physical_scale_boxes uses those as square crop sides, not min(H,W). CLIPLoss.forward adds motif_scale_weight * per-scale text loss when fracs are set; last_motif_scale_losses is logged onto the AdaptiveAdam dataset as clip_motif_* columns. Preset venice_250214_motif_scale does not change venice_250214().to_venice_golden()==GOLDEN. Tests: geometry + additive CLIP path (36 passed in the targeted file; full suite 238 passed, 8 skipped). This is not silhouette matching and not a saved multi-scale look yet — the gate (repeating skeleton language at two sizes on a saved look) is still open.
