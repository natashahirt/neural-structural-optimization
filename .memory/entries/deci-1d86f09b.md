---
id: deci-1d86f09b
type: decision
project: semantopology_hardfork
parent_id: plan-335b610c
title: Selected Stage 6 look is occupancy init plus 4000→400
node_label: Selected Stage 6 look is occupancy init plus 4000→
tags: stage-6,sketch,selected-look,weight-end-400
status: active
open_threads: 0
success: 'null'
files: script/resources/results/stage6_sketch12_init_anneal/summary.json@dbb936b,
  script/resources/results/stage6_sketch12_init_anneal/comparison.png@fc94a88, script/resources/results/stage6_sketch12_weight_end_800/summary.json@dbb936b,
  script/resources/results/stage6_sketch12_weight_end_1200/summary.json@dbb936b, script/resources/results/stage6_sketch12_motif_recurrence/summary.json@23587d6,
  script/resources/results/stage6_sketch12_patch_vocabulary/summary.json@fc94a88,
  script/resources/results/stage6_sketch12_patch_vocabulary_strong/summary.json@fc94a88
session_id: sess-8765f4fa
created_at: '2026-09-05T04:10:39.798682+00:00'
updated_at: '2026-09-05T04:10:39.798682+00:00'
has_child_rationale: visual comparison plus compliance/CLIP tradeoff
results: '[{"metric": "compliance", "value": 83.52705383300781, "unit": "raw", "split":
  "CLIP-on-sketch12-global-end-400", "window": "final-step", "criterion": "selected-look",
  "source": "script/resources/results/stage6_sketch12_init_anneal/summary.json"},
  {"metric": "clip_loss", "value": 323.8759765625, "unit": "raw", "split": "CLIP-on-sketch12-global-end-400",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_init_anneal/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.5451500298792412, "unit": "fraction",
  "split": "CLIP-on-sketch12-global-end-400", "window": "final-step", "source": "script/resources/results/stage6_sketch12_init_anneal/summary.json"},
  {"metric": "compliance", "value": 87.44351196289062, "unit": "raw", "split": "CLIP-on-sketch12-global-end-800",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_800/summary.json"},
  {"metric": "compliance", "value": 90.60646057128906, "unit": "raw", "split": "CLIP-on-sketch12-global-end-1200",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_weight_end_1200/summary.json"},
  {"metric": "compliance", "value": 88.03839111328125, "unit": "raw", "split": "CLIP-on-sketch12-patch-1600-to-800",
  "window": "final-step", "source": "script/resources/results/stage6_sketch12_patch_vocabulary_strong/summary.json"}]'
---
Human judgment on the CLIP-on sketch-12 comparison panels: occupancy init plus stage-tied spatial anneal 4000→400 is the selected look. Raising the final spatial weight to 800 or 1200, then adding Gram/autocorrelation motif or patch-vocabulary recurrence, thickened the V and raised compliance without making the drawing's local language recur. CLIP skeletons also look cleaner at 400. Treat motif_weight and patch_weight as off-by-default research knobs, not as the Stage 6 default.
