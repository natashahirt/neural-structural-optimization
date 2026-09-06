---
id: deci-2b042aec
type: decision
project: semantopology_hardfork
parent_id: plan-073cebb0
title: Dream CLIP sees the whole building, not storey crops
node_label: Dream CLIP sees the whole building, not storey cro
tags: e1,clip-dream,whole-elevation,decision
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-0ff8a38f
created_at: '2026-09-06T04:48:41.822425+00:00'
updated_at: '2026-09-06T04:48:41.822425+00:00'
has_child_rationale: User asked to dream CLIP the whole building so load paths can
  pass through storeys.
---
E1 dream CLIP is the whole elevation (Venice RandomResizedCrop only). Motif-scale storey/member crops are off. Neutral init does not raise load-site floors to 1 during the dream (they are unioned when the mass prior is applied). Occupancy is that one upsampled drawing, rank-thresholded, no storey distance envelope.

The first dream (clip_dream_layout_butterfly_wing_venation) used motif-scale crops plus floor collectors; ink almost never crossed a floor (0/1/1 of 32 columns). That cannot author a through-building load path.
