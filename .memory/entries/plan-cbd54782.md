---
id: plan-cbd54782
type: plan
project: semantopology_hardfork
parent_id: plan-335b610c
title: Emphasize recurrence with patch-vocabulary matching
node_label: Emphasize recurrence with patch-vocabulary matchin
tags: stage-6,sketch,motif,patch-vocabulary,plan
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-a11a6a0c
created_at: '2026-09-05T01:34:20.134981+00:00'
updated_at: '2026-09-05T01:34:20.134981+00:00'
---
Try a translation-invariant patch-vocabulary motif loss because global Gram/autocorrelation statistics reduced the scalar loss but produced only a subtle visible change. Keep CLIP on and global spatial annealing at 4000→800. Build the vocabulary only from nonblank sketch occupancy patches, never load-only collector rows; use differentiable soft nearest-neighbor matching in both directions; keep it off coarse, stronger after the first upsample, and moderate at full resolution. Require translated-motif tolerance, repeated-column rejection, finite gradients, no-op parity, a green full suite, and one CLIP-on sketch-12 comparison against the fixed-statistics run.
