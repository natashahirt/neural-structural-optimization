---
id: deci-cc0dde74
type: decision
project: semantopology_hardfork
parent_id: plan-1c9a2525
title: 'Sketch gravity path: occupancy union loaded pixels, CLIP stays'
node_label: 'Sketch gravity path: occupancy union loaded pixels'
tags: stage-6,sketch,load-pixels,decision
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-7aba5a79
created_at: '2026-09-04T22:55:48.575523+00:00'
updated_at: '2026-09-04T22:55:48.575523+00:00'
---
Refined Stage 6 prior: penalize mass outside occupancy ∪ load-application pixels, not occupancy alone.

Loads on void are catastrophically compliant, so a sketch that forbids those pixels forces the optimizer back to the default frame (floors/columns). Extra-allowing exactly the elements that correspond to nodes with nonzero Fx or Fy removes that fight. Collectors at the loads are free; between them the drawing is the allowed path. Same rule for gravity floors and lateral point loads. No extra band — one element per loaded node. CLIP stays on.

Look: Venice 250214 + sketch 12 with this template.
