---
id: note-59c278ce
type: note
project: semantopology_hardfork
parent_id: todo-a37ba202
title: E2 draft runner path
node_label: E2 draft runner path
tags: e2,cnn,parked,note
status: active
open_threads: 0
success: 'null'
files: script/cnn_motif_scale.py@d9c9d07
session_id: sess-c5f62176
created_at: '2026-09-06T04:18:06.870861+00:00'
updated_at: '2026-09-06T04:18:06.870861+00:00'
---
Draft runner lives at script/cnn_motif_scale.py (CNN at full 128x256, Adam, Venice CLIP algebra, no resolution ladder). Needs lr calibration (structure-only) before any CLIP run. Frozen-decoder (latent-only) flag not added yet. Do not start until E1 is judged.
