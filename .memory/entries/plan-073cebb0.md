---
id: plan-073cebb0
type: plan
project: semantopology_hardfork
parent_id: plan-3d516df5
title: 'E1: CLIP dreams, physics builds (load-path prior)'
node_label: 'E1: CLIP dreams, physics builds (load-path prior)'
tags: clip-scale,e1,clip-dream,load-path,scaffold,plan
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-c5f62176
created_at: '2026-09-06T04:04:57.220192+00:00'
updated_at: '2026-09-06T04:04:57.220192+00:00'
has_child_rationale: User asked to build E1 now and park E2/E3; this is the load-path
  experiment.
---
CLIP motif should inform the *load path*, not decorate a physics-chosen frame. Stage 6 sketches already do this via a spatial mass prior on physical density. The failed layout-distillation student (clip_motif_layout_no_occupancy) reused that prior but the scaffold was tautological: the teacher was itself a physics+CLIP run, so its ink already sat on the load path. Measured on that run: scaffold mean 0.7668 (Stage 6 templates are 0.64–0.83, so sparsity is not the difference), spatial_mass_loss 0.0078 vs Stage 6 residuals 0.31–0.60, mass_on_allowed 0.9994.

E1 (this plan): CLIP-only dream at the coarse 32×64 grid (no FEA), upsample, extract a storey/member scaffold with ink threshold binary-searched to allowed-mean ≈0.75, gate that a physics-layout proxy scores spatial_mass_loss ≥0.25 against it, then run AdaptivePixel + Venice CLIP + Stage 6 anneal 4000→400 with init_from_occupancy. Do not seed from a physics teacher.

Success: load path visibly differs from the no-prior three-bay frame and traces to the dream motif; compliance ≤85 (Stage 6 paid 73–84 vs ~72). Gate fail is a stop, not a full run.

Default prompt: butterfly wing venation (last encouraging CLIP look). Runner: script/clip_dream_layout.py.
