---
id: todo-8f888aea
type: todo
project: semantopology_hardfork
parent_id: deci-640e26f0
title: 'Stage 8 first gate: fix CNN-pixel handoff volume'
node_label: 'Stage 8 first gate: fix CNN-pixel handoff volume'
tags: stage-8,handoff,todo
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-a469ebb9
created_at: '2026-09-04T18:42:48.122755+00:00'
updated_at: '2026-09-05T06:12:50.462005+00:00'
related_to: plan-6fd637e5
---
When Stage 8 starts: fix `constrained_logits` so one filter+projection lands on volfrac under Heaviside before designing CNN+pixel residual. Do not beta-continue or rely on MMA/OC. Pins: expe-38a0e2f4, note-7dabd6e1, CanonicalHandoffVolumeTest.
