---
id: expe-b47d41cd
type: experiment
project: semantopology_hardfork
parent_id: plan-e1a43a00
title: Occupancy-init 4000→400 replicates on sketches 1, 3, 6
node_label: 'Occupancy-init 4000→400 replicates on sketches 1, '
tags: stage-6,sketch,replication,skeletons,clip-on
status: active
open_threads: 0
success: 'true'
files: script/resources/results/stage6_sketch1_init_anneal/summary.json@73c23fd, script/resources/results/stage6_sketch3_init_anneal/summary.json@73c23fd,
  script/resources/results/stage6_sketch6_init_anneal/summary.json@73c23fd
session_id: sess-60a28823
created_at: '2026-09-05T05:42:52.040796+00:00'
updated_at: '2026-09-05T05:42:52.040796+00:00'
results: '[{"metric": "compliance", "value": 81.90522003173828, "unit": "raw", "split":
  "CLIP-on-sketch1-global-end-400", "window": "final-step", "source": "script/resources/results/stage6_sketch1_init_anneal/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.3627654065234407, "unit": "fraction",
  "split": "CLIP-on-sketch1-global-end-400", "window": "final-step", "source": "script/resources/results/stage6_sketch1_init_anneal/summary.json"},
  {"metric": "compliance", "value": 73.38639068603516, "unit": "raw", "split": "CLIP-on-sketch3-global-end-400",
  "window": "final-step", "source": "script/resources/results/stage6_sketch3_init_anneal/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.4960772008824873, "unit": "fraction",
  "split": "CLIP-on-sketch3-global-end-400", "window": "final-step", "source": "script/resources/results/stage6_sketch3_init_anneal/summary.json"},
  {"metric": "compliance", "value": 75.28414916992188, "unit": "raw", "split": "CLIP-on-sketch6-global-end-400",
  "window": "final-step", "source": "script/resources/results/stage6_sketch6_init_anneal/summary.json"},
  {"metric": "mass_on_occupancy", "value": 0.4542682564824302, "unit": "fraction",
  "split": "CLIP-on-sketch6-global-end-400", "window": "final-step", "source": "script/resources/results/stage6_sketch6_init_anneal/summary.json"}]'
---
Human judgment after CLIP-on occupancy-init 4000→400 replication: the selected Stage 6 recipe generalizes beyond sketch 12. CLIP prompt remains skeletons. Sketches 1, 3, and 6 finished with the drawing as primary structure rather than the gothic CLIP basin; 9 and 11 were still running. Sketch 3 reached compliance 73.39 (below the no-sketch golden ~74) with mass-on-occupancy 0.496. Do not reintroduce motif/patch for this look.
