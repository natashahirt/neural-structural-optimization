---
id: expe-cc2f2cee
type: experiment
project: semantopology_hardfork
parent_id: plan-1c9a2525
title: 'Look: Venice 250214 + sketch 12 still looks like CLIP, not the V'
node_label: 'Look: Venice 250214 + sketch 12 still looks like C'
tags: stage-6,sketch,look,venice-250214
status: active
open_threads: 0
success: 'true'
files: ''
session_id: sess-7aba5a79
created_at: '2026-09-04T22:43:33.508517+00:00'
updated_at: '2026-09-04T23:39:47.742907+00:00'
results: '[{"metric": "compliance", "value": 75.692, "split": "venice-250214+sketch12",
  "window": "113-steps-converged", "criterion": "look-not-parity", "source": "stage6_sketch_on_golden"},
  {"metric": "clip_loss", "value": 287.931, "split": "venice-250214+sketch12", "window":
  "113-steps-converged", "source": "stage6_sketch_on_golden"}, {"metric": "mean_physical_density",
  "value": 0.3002, "split": "venice-250214+sketch12", "window": "113-steps-converged",
  "criterion": "volfrac=0.3", "source": "stage6_sketch_on_golden"}, {"metric": "mass_on_occupancy",
  "value": 0.3089, "split": "venice-250214+sketch12", "window": "113-steps-converged",
  "criterion": ">occupancy-frac~0.22", "source": "stage6_sketch_on_golden"}, {"metric":
  "volume_actual", "value": 0.3186, "split": "venice-250214+sketch12", "window": "113-steps-converged",
  "source": "stage6_sketch_on_golden"}, {"metric": "steps", "value": 113, "split":
  "venice-250214+sketch12", "window": "converged", "source": "stage6_sketch_on_golden"}]'
related_to: expe-416235a9
---
Look run: Venice 250214 knobs (CLIP skeletons, AdaptivePixel, same init image/seed/schedule) plus sketch 12 occupancy at weight 400.

Converged in 113 steps (upsamples at 48, 69) vs reference 124. Mean physical density 0.300 (volfrac held). Mass on sketch-12 occupancy 0.309 vs occupancy covering ~0.22 of cells, so some redistribution. Compliance 75.69 vs golden 73.997; CLIP 287.93 vs 283.37. Total 640 includes the sketch term.

Visual: the design still looks like the golden gothic/skeleton drawing, not like sketch 12's V. CLIP still owns the silhouette at this weight. Occupancy prior is doing work in the density budget, not rewriting the CLIP image.

Panels: script/resources/results/stage6_sketch12_on_golden/{occupancy_vs_reference,comparison,density_vs_reference,sketch_run,physical_density}.png
