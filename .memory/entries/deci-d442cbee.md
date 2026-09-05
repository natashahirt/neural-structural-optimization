---
id: deci-d442cbee
type: decision
project: semantopology_hardfork
parent_id: plan-3d516df5
title: Restore raw motif guidance; separate decoration from layout
node_label: Restore raw motif guidance; separate decoration fr
tags: clip-scale,decision,raw-motif,structural-layout,two-channel
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/models/loss_clip.py@9816779, neural_structural_optimization/models/model_base.py@9816779,
  neural_structural_optimization/tests/test_clip_motif_scale.py@9816779, script/venice_golden_250214.py@9816779
session_id: sess-1e640c29
created_at: '2026-09-05T19:57:48.880424+00:00'
updated_at: '2026-09-05T19:57:48.880424+00:00'
invalidates: expe-083c1ce7
---
User judged the raw-input no-occupancy motif look substantially more compelling than the physical-density replacement. Revert commit 9816779's routing/output code while preserving both experiment artifact directories for comparison.

Clarified target: the motif has two distinct jobs. Local/raw CLIP should retain decorative skull/teeth/rib vocabulary. Separately, a motif-derived spatial layout force should reorganize where structural density sits, analogous to how Stage 6 sketch occupancy reorganized density. Feeding filtered density directly to the decorative crop loss conflated those jobs and traded away the stronger semantic result. Next design should keep the raw motif path and add—not substitute—a coarse motif-layout prior.
