---
id: plan-e1a43a00
type: plan
project: semantopology_hardfork
parent_id: plan-335b610c
title: Replicate 4000→400 occupancy init on the sketch corpus
node_label: Replicate 4000→400 occupancy init on the sketch co
tags: stage-6,sketch,replication,corpus
status: active
open_threads: 2
success: 'null'
files: ''
session_id: sess-8765f4fa
created_at: '2026-09-05T04:11:26.252586+00:00'
updated_at: '2026-09-05T05:57:28.761549+00:00'
related_to: plan-039cabee
---
Leave patch-vocabulary and Gram motif terms off. Replicate the selected Stage 6 recipe (occupancy ∪ load pixels, init_from_occupancy, spatial anneal 4000→400, CLIP on) across the rest of the Stage 6 corpus (sketches 1, 3, 6, 9, 11; 12 is the existing selected look). Judge replication by whether the drawing's global occupancy becomes the primary structure versus the gothic CLIP basin, plus compliance, CLIP loss, and mass-on-occupancy.
