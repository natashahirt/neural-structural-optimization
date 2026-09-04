---
id: expe-38a0e2f4
type: experiment
project: semantopology_hardfork
parent_id: plan-13117c2b
title: CNN handoff still misses volfrac; sign still flips with beta
node_label: CNN handoff still misses volfrac; sign still flips
tags: stage-4,stage-8,handoff,re-measure,projection
status: active
open_threads: 0
success: 'false'
files: neural_structural_optimization/train/utils.py@d6106bb, neural_structural_optimization/tests/test_discretization.py@99e3ffa,
  neural_structural_optimization/structural/api.py@99e3ffa
session_id: sess-a469ebb9
created_at: '2026-09-04T18:32:41.024598+00:00'
updated_at: '2026-09-04T18:32:58.355934+00:00'
has_child_rationale: Stage 4 promised a re-measure of the CNN handoff before anyone
  designs Stage 8 around old numbers.
results: '[{"metric": "handoff_volume_error", "value": -6.4, "unit": "%", "split":
  "crane 32x32 volfrac=0.3 eta=0.5 beta=4 RandomState(0)*2", "window": "design-region-mean
  of reprojected handoff", "criterion": "should equal volfrac", "source": "CanonicalHandoffVolumeTest"},
  {"metric": "handoff_volume_error", "value": 11.2, "unit": "%", "split": "crane 32x32
  volfrac=0.3 eta=0.5 beta=16 RandomState(0)*2", "window": "design-region-mean of
  reprojected handoff", "criterion": "should equal volfrac", "source": "CanonicalHandoffVolumeTest"},
  {"metric": "handoff_volume_error", "value": -7.1, "unit": "%", "split": "multistory_building
  60x60 interval=16 volfrac=0.3 eta=0.5 beta=4 RandomState(0)*2", "window": "mean
  of reprojected handoff", "criterion": "should equal volfrac", "source": "stage-4-remeasure"},
  {"metric": "handoff_volume_error", "value": 5.9, "unit": "%", "split": "multistory_building
  60x60 interval=16 volfrac=0.3 eta=0.5 beta=16 RandomState(0)*2", "window": "mean
  of reprojected handoff", "criterion": "should equal volfrac", "source": "stage-4-remeasure"}]'
related_to: note-7dabd6e1
---
Re-measured after Stage 4 slice A (`Environment.render` now uses cone_filter=True; `constrained_logits` still cone_filter=False).

Protocol: RandomState(0).randn * 2 logits, volfrac=0.3, eta=0.5, heavyside=True. Handoff = physical_density(vc=True, cone_filter=False). After = physical_density(handoff, vc=False, cone_filter=True), design-region mean.

crane 32x32 matches note-7dabd6e1 exactly: after_dmean 0.28082 (-6.4%) at beta=4 and 0.33348 (+11.2%) at beta=16. The sign still flips with beta. Slice A did not reshape this defect; Stage 8 remains blocked on it.

multistory_building 60x60 (interval=16, same protocol as the render pin): after mean 0.27865 (-7.1%) at beta=4 and 0.31770 (+5.9%) at beta=16. Note-7dabd6e1 quoted 0.27750 / 0.32553 on a slightly different setup; the qualitative gap is unchanged.
