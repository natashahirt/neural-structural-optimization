---
id: expe-83fd934e
type: experiment
project: semantopology_hardfork
parent_id: plan-335b610c
title: CLIP-on sketch-12 motif recurrence prototype
node_label: CLIP-on sketch-12 motif recurrence prototype
tags: stage-6,sketch,motif,recurrence,clip-on,passed-tests
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/models/loss_sketch.py@23587d6, neural_structural_optimization/models/model_base.py@23587d6,
  neural_structural_optimization/models/__init__.py@23587d6, neural_structural_optimization/experiment.py@23587d6,
  neural_structural_optimization/tests/test_sketch.py@23587d6, script/stage6_sketch_on_golden.py@23587d6,
  script/resources/results/stage6_sketch12_motif_recurrence/summary.json@23587d6,
  script/resources/results/stage6_sketch12_motif_recurrence/comparison_summary.json@23587d6,
  script/resources/results/stage6_sketch12_motif_recurrence/comparison.png@23587d6
session_id: sess-a11a6a0c
created_at: '2026-09-05T01:19:05.738014+00:00'
updated_at: '2026-09-05T01:19:05.738014+00:00'
results: '[{"metric": "mass_on_occupancy", "value": 0.5451500298792412, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-400", "window": "final-step", "source": "stage6_sketch12_init_anneal/summary.json"},
  {"metric": "compliance", "value": 83.52705383300781, "unit": "raw", "split": "CLIP-on-sketch12-global-end-400",
  "window": "final-step", "source": "stage6_sketch12_init_anneal/summary.json"}, {"metric":
  "mass_on_occupancy", "value": 0.5712354885255695, "unit": "fraction", "split": "CLIP-on-sketch12-global-end-800",
  "window": "final-step", "source": "stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "compliance", "value": 87.44351196289062, "unit": "raw", "split": "CLIP-on-sketch12-global-end-800",
  "window": "final-step", "source": "stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "motif_loss", "value": 0.007134634535759687, "unit": "MSE", "split":
  "CLIP-on-sketch12-global-end-800", "window": "final-step", "source": "stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.5915339490133035, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-1200", "window": "final-step", "source": "stage6_sketch12_weight_end_1200/summary.json"},
  {"metric": "compliance", "value": 90.60646057128906, "unit": "raw", "split": "CLIP-on-sketch12-global-end-1200",
  "window": "final-step", "source": "stage6_sketch12_weight_end_1200/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.5722075662228328, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-800-plus-motif", "window": "final-step", "source":
  "stage6_sketch12_motif_recurrence/summary.json"}, {"metric": "compliance", "value":
  87.34049224853516, "unit": "raw", "split": "CLIP-on-sketch12-global-end-800-plus-motif",
  "window": "final-step", "source": "stage6_sketch12_motif_recurrence/summary.json"},
  {"metric": "motif_loss", "value": 0.005623123608529568, "unit": "MSE", "split":
  "CLIP-on-sketch12-global-end-800-plus-motif", "window": "final-step", "source":
  "stage6_sketch12_motif_recurrence/summary.json"}]'
---
Implemented a separate translation-invariant motif prior using fixed horizontal/vertical/diagonal edge filters, normalized orientation Gram matrices, and local axial/diagonal autocorrelations at scales 1/2/4. Blank reference regions contribute zero edge energy; the model conditions motif statistics on occupancy only, so load-only collector rows remain spatial allowances rather than reference motifs. AdaptivePixel schedule is 0 on coarse, peak after first resize, moderate at full resolution. Defaults are fully off. A synthetic repeated-column field scored far worse than a translated V reference, satisfying the prototype guardrail. Full suite: 219 passed, 8 skipped, 241 subtests. In CLIP-on sketch 12, weight_end=800 was selected over 1200 as the global tradeoff. The motif run used global 4000→800 and motif 0→8000→3000; it reduced final motif loss while preserving mass-on-sketch and compliance relative to the global-800 run. Visual and scalar comparison is saved under stage6_sketch12_motif_recurrence.
