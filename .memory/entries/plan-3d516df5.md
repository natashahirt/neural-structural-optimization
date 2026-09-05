---
id: plan-3d516df5
type: plan
project: semantopology_hardfork
parent_id: plan-039cabee
title: 'CLIP motif scale: skeletons at more than one physical length'
node_label: 'CLIP motif scale: skeletons at more than one physi'
tags: clip-scale,skeletons,plan,not-silhouette
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-60a28823
created_at: '2026-09-05T05:54:45.885125+00:00'
updated_at: '2026-09-05T06:12:49.452957+00:00'
related_to: plan-6fd637e5
---
Own question, not Stage 7 blending and not silhouette matching. CLIP gothic/skeleton language locks to one bay size because Venice resizes short side to 512 then RandomResizedCrop scale (224/480, 1.0). AdaptivePixel grid changes do not change motif size. Lesson from Stage 6: occupancy succeeded by not fighting CLIP silhouette; JPEG-to-JPEG / Gram / patch failed. This stage asks the text prompt skeletons to fire at more than one physical length scale (whole building / storey / member) on a second CLIP path. Venice 250214 path stays frozen as A/B baseline. Keep occupancy init + 4000 to 400. Out: image-to-image CLIP, sketch patch/Gram, hard mask, blending controller, CNN handoff. Gate: repeating skeleton language at two clearly different sizes; volfrac holds; compliance in the selected-look band.
