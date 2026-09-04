---
id: plan-5f58bc55
type: plan
project: semantopology_hardfork
parent_id: plan-1c9a2525
title: 'Stage 6: occupancy init + sketch-weight anneal'
node_label: 'Stage 6: occupancy init + sketch-weight anneal'
tags: stage-6,sketch,init,anneal,plan
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-1c5b4d01
created_at: '2026-09-04T23:54:26.710743+00:00'
updated_at: '2026-09-04T23:54:26.710743+00:00'
---
Stage 6 follow-up: occupancy init + stage-tied weight anneal so the drawing is the starting basin and compliance can refine it. CLIP stays on for the look (Venice 250214 + sketch 12). Template remains occupancy union load pixels. Init from the allowed map at the coarse grid (overwrite thick_outer_lins). Anneal by AdaptivePixel stage: 4000 at 32x64, midpoint at 64x128, 400 at 128x256 (CLIP weight is compliance*10, ~4900 at step 0). Venice-without-sketch stays a no-op (path=None, init_from_occupancy=False, weight_end=None). Hard mask and CLIP-off look stay out of slice.
