---
id: expe-a156a5ba
type: experiment
project: semantopology_hardfork
parent_id: plan-335b610c
title: CLIP-on sketch-12 motif recurrence prototype (pinned results)
node_label: CLIP-on sketch-12 motif recurrence prototype (pinn
tags: stage-6,sketch,motif,recurrence,clip-on,passed-tests
status: active
open_threads: 0
success: 'true'
files: script/resources/results/stage6_sketch12_init_anneal/summary.json@dbb936b,
  script/resources/results/stage6_sketch12_weight_end_800/summary.json@dbb936b, script/resources/results/stage6_sketch12_weight_end_1200/summary.json@dbb936b,
  script/resources/results/stage6_sketch12_motif_recurrence/summary.json@23587d6,
  script/resources/results/stage6_sketch12_motif_recurrence/comparison_summary.json@23587d6
session_id: sess-a11a6a0c
created_at: '2026-09-05T01:19:27.371456+00:00'
updated_at: '2026-09-05T01:19:27.371456+00:00'
invalidates: expe-83fd934e
results: '[{"metric": "mass_on_occupancy", "value": 0.5451500298792412, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-400", "window": "final-step", "source": "script/resources/results/stage6_sketch12_init_anneal/summary.json"},
  {"metric": "compliance", "value": 83.52705383300781, "unit": "raw", "split": "CLIP-on-sketch12-global-end-400",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_init_anneal/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.5712354885255695, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-800", "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "compliance", "value": 87.44351196289062, "unit": "raw", "split": "CLIP-on-sketch12-global-end-800",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "motif_loss", "value": 0.007134634535759687, "unit": "MSE", "split":
  "CLIP-on-sketch12-global-end-800", "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.5915339490133035, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-1200", "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_1200/summary.json"},
  {"metric": "compliance", "value": 90.60646057128906, "unit": "raw", "split": "CLIP-on-sketch12-global-end-1200",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_1200/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.5722075662228328, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-800-plus-motif", "window": "final-step", "source":
  "script/resources/results/stage6_sketch12_motif_recurrence/summary.json"}, {"metric":
  "compliance", "value": 87.34049224853516, "unit": "raw", "split": "CLIP-on-sketch12-global-end-800-plus-motif",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_motif_recurrence/summary.json"},
  {"metric": "motif_loss", "value": 0.005623123608529568, "unit": "MSE", "split":
  "CLIP-on-sketch12-global-end-800-plus-motif", "window": "final-step", "source":
  "script/resources/results/stage6_sketch12_motif_recurrence/summary.json"}]'
---
Pinned correction of the Stage 6 motif experiment record. The fixed-filter Gram/autocorrelation prototype passed its repeated-column negative case and the full suite (219 passed, 8 skipped). The 400/800/1200 global sweep selected 800 as the better mass-versus-compliance tradeoff. The CLIP-on motif run used global 4000→800 and motif 0→8000→3000, lowering final motif loss while holding mass-on-sketch and compliance essentially level against global 800. See the pinned comparison summary for all run conditions and metrics.
