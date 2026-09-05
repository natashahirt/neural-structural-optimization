---
id: plan-cbdb1948
type: plan
project: semantopology_hardfork
parent_id: plan-3d516df5
title: Motif layout self-distillation
node_label: Motif layout self-distillation
tags: clip-scale,motif-layout,self-distillation,scaffold,plan
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/models/loss_sketch.py@168499b, neural_structural_optimization/experiment.py@9ab1887,
  neural_structural_optimization/tests/test_sketch.py@168499b, neural_structural_optimization/tests/test_clip_motif_scale.py@9816779,
  neural_structural_optimization/tests/test_experiment.py@54957c4, script/motif_layout_distillation.py@ffc8846
session_id: sess-dbe347bd
created_at: '2026-09-05T20:43:35.845351+00:00'
updated_at: '2026-09-05T20:43:35.845351+00:00'
has_child_rationale: User confirmed the motif-layout distillation plan; implementation
  starts now.
---
Two-pass self-distillation: a raw multi-scale CLIP teacher produces the compelling motif look; a deterministic soft multiscale occupancy envelope is extracted from teacher ink at physically derived building/storey/member scales; a student initializes from the teacher, keeps raw CLIP decoration, and anneals sketch_mass_prior_loss on that scaffold (4000→400) so members move without feeding filtered density into decorative CLIP.

Empty layout config stays bit-identical to Venice. Not a user sketch. Do not overwrite clip_motif_scale_no_occupancy or clip_motif_scale_physical_density_no_occupancy. Student gate vs teacher expe-2293aefa: recognizable raw motif language, major physical-density members on the scaffold, mean density 0.30, compliance not worse than 80.04.
