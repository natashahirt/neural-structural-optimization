---
id: chec-dfb96ffa
type: checkpoint
project: semantopology_hardfork
parent_id: plan-073cebb0
title: E1 runner and tautology gate implemented
node_label: E1 runner and tautology gate implemented
tags: e1,clip-dream,implementation,checkpoint
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/models/loss_sketch.py@e6005e5, script/clip_dream_layout.py@e6005e5,
  neural_structural_optimization/tests/test_sketch.py@e6005e5, neural_structural_optimization/tests/test_clip_motif_scale.py@e6005e5
session_id: sess-c5f62176
created_at: '2026-09-06T04:18:04.475691+00:00'
updated_at: '2026-09-06T04:18:04.475691+00:00'
---
E1 runner and gate are implemented, not yet executed as a CLIP look.

- `motif_layout_threshold_for_allowed_mean` binary-searches the ink cut so scaffold.mean() ~ 0.75 (Stage 6 allowed-area band).
- `scaffold_spatial_mass_loss` / `evaluate_tautology_gate` require spatial_mass_loss >= 0.25 vs a physics-layout proxy (default: clip_motif_layout_no_occupancy/teacher/physical_density.npy) before any AdaptiveAdam spend.
- `apply_scaffold_as_occupancy_prior` uses Stage 6 occupancy union load-site init + 4000->400 anneal, not teacher-z init.
- Runner: `PYTHONPATH=$PWD python script/clip_dream_layout.py --dream-only` then drop `--dream-only` if the gate passes. Default prompt: butterfly wing venation.

19 unit tests passed (ClipDreamScaffoldGateTest, ClipDreamLayoutDirTest, MotifLayoutScaffoldTest). Full CLIP dream not run this session.
