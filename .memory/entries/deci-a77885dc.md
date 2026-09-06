---
id: deci-a77885dc
type: decision
project: semantopology_hardfork
parent_id: plan-cbdb1948
title: Physics+CLIP teachers make a tautological scaffold
node_label: Physics+CLIP teachers make a tautological scaffold
tags: clip-scale,scaffold,tautology,decision
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-c5f62176
created_at: '2026-09-06T04:05:06.254436+00:00'
updated_at: '2026-09-06T04:05:18.315073+00:00'
has_child_rationale: Explains why the existing teacher/student path was vacuous; E1
  is the successor.
results: '[{"metric": "scaffold_mean", "value": 0.7668, "split": "clip_motif_layout_no_occupancy",
  "source": "summary.json"}, {"metric": "spatial_mass_loss", "value": 0.0078, "split":
  "clip_motif_layout_no_occupancy", "criterion": ">=0.25 for a non-tautological prior",
  "source": "summary.json"}, {"metric": "mass_on_allowed", "value": 0.9994, "split":
  "clip_motif_layout_no_occupancy", "source": "summary.json"}]'
related_to: plan-073cebb0
---
The layout-distillation student did not fail because CLIP cannot inform layout. It failed because the teacher had already seen compliance, so the extracted scaffold endorsed the physics load path. On clip_motif_layout_no_occupancy: scaffold_mean=0.7668, spatial_mass_loss=0.0078, mass_on_allowed=0.9994. Stage 6 sketch templates have similar allowed area (0.64–0.83) but spatial_mass_loss 0.31–0.60 — they fought physics. Fix: the scaffold must come from a field that has never seen FEA, then a pre-run tautology gate (spatial_mass_loss ≥0.25 vs a physics-layout proxy) before spending a student run.
