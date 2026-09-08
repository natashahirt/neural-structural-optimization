---
id: note-6f89ba2e
type: note
project: semantopology_hardfork
parent_id: deci-3055b18d
title: Stage 6 restore, occupancy absolute, deferred critic items
node_label: Stage 6 restore, occupancy absolute, deferred crit
tags: clip,critic,stage6,deferred
status: active
open_threads: 0
success: 'null'
files: script/stage6_sketch_on_golden.py@a91aa20, script/clip_saliency_compliance.py@a91aa20,
  script/clip_dream_layout.py@a91aa20
session_id: sess-bd84de5d
created_at: '2026-09-08T17:45:28.643695+00:00'
updated_at: '2026-09-08T17:45:28.643695+00:00'
has_child_rationale: Critic P1s fixed vs items still deferred
---
Critic CONCERN (not FAIL) on run-bf46a812. Accepted P1s were fixed in the same run: 48 tracked Stage 6 files restored under `SUCCESS_sketch_to_structure` (git R100); corpus strip retargeted so a missing sketch raises FileNotFoundError; occupancy PNG no longer min-max stretched; saliency now passes occupancy into `report_design_metrics`; Stage 6 writes design `.npy` via `save_design_arrays`.

Still deferred, not blocking this checkpoint:
- dream float32 cast in `clip_dream_layout`
- folded-arm abort
- saliency scaffold metrics remain null (occupancy is now reported; scaffold keys were not filled)

Do not treat SDS FrozenDenoiser as a diffusion prior; it stays opt-in only.
