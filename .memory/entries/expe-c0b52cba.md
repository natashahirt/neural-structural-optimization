---
id: expe-c0b52cba
type: experiment
project: semantopology_hardfork
parent_id: plan-073cebb0
title: 'Whole-building CLIP dream: gate miss 0.192 vs 0.25'
node_label: 'Whole-building CLIP dream: gate miss 0.192 vs 0.25'
tags: e1,clip-dream,whole-elevation,experiment,gate
status: active
open_threads: 0
success: 'false'
files: ''
session_id: sess-0ff8a38f
created_at: '2026-09-06T05:06:47.334729+00:00'
updated_at: '2026-09-06T05:06:47.334729+00:00'
results: '[{"metric": "spatial_mass_loss", "value": 0.1918, "split": "whole-elevation-dream-vs-layout-teacher-density",
  "criterion": ">=0.25", "source": "script/resources/results/clip_dream_layout_whole_butterfly_wing_venation/summary.json"},
  {"metric": "scaffold_mean", "value": 0.75, "split": "whole-elevation-dream-vs-layout-teacher-density",
  "source": "script/resources/results/clip_dream_layout_whole_butterfly_wing_venation/summary.json"},
  {"metric": "mass_on_scaffold", "value": 0.8018, "split": "whole-elevation-dream-vs-layout-teacher-density",
  "source": "script/resources/results/clip_dream_layout_whole_butterfly_wing_venation/summary.json"}]'
---
Whole-elevation CLIP dream (no motif-scale crops, no load-site frames), 64 steps, prompt butterfly wing venation. Occupancy is rank-thresholded one-image (mean 0.75, threshold 0.25). Gate vs clip_motif_layout_no_occupancy/teacher/physical_density.npy: spatial_mass_loss=0.1918, criterion >=0.25, failed. mass_on_scaffold=0.8018. Physics not run.

Visually this is one elevation (venation/wing through the height), unlike the previous four-storey-framed dream. The gate miss is narrow vs the old tautological 0.0078. A sparser occupancy (lower target_mean) would make the prior fight physics harder; --skip-gate would spend the Stage 6 recipe on this map as-is.
