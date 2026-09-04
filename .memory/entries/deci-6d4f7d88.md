---
id: deci-6d4f7d88
type: decision
project: semantopology_hardfork
parent_id: plan-e9993b76
title: Sketch guidance is a spatial prior on where the volume budget sits
node_label: Sketch guidance is a spatial prior on where the vo
tags: stage-6,sketch,material-distribution,decision
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-cf4626a3
created_at: '2026-09-04T19:54:35.849312+00:00'
updated_at: '2026-09-04T20:45:46.331884+00:00'
has_child_rationale: Confirmed research target for Stage 6, after Stage 5 experiment
  config.
related_to: plan-c9c6884b,deci-640e26f0,plan-1c9a2525
---
User confirmed Stage 6 sketch guidance as a spatial prior on material distribution, not silhouette matching and not CLIP.

The 30% volfrac budget stays global. A cleaned occupancy map from the sketch (threshold so grids/ghost construction drop out) says WHERE that budget is encouraged: dark/hatched/busy regions get more material; paper stays closer to void. Physics still invents members; CLIP (text) remains a separate semantic term. No JPEG-to-JPEG copy, no hatch-as-structure, no image-to-image CLIP for these drawings.

Corpus that drove this: semantopology_venice/resources/input_images/sketches/{1,3,6,9,11,12}.jpg — disconnected islands (1), loops/negative space (3), fine twigs vs mass (6), faint grid (9), heavy V vs light rectangles (11), gestural path (12).

Implementation direction when Stage 6 starts: preprocess to inspectable occupancy; pull canonical physical density toward that map; keep volfrac; optional weak edge term later if forks wash out.
