---
id: todo-47c9d3de
type: todo
project: semantopology_hardfork
parent_id: deci-ba13459e
title: 'guards: fix double-optimize, swallowed clip_weight, expose fix_right_wall'
node_label: 'guards: fix double-optimize, swallowed clip_weight'
tags: ''
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-2b965d62
created_at: '2026-09-04T13:54:46.187451+00:00'
updated_at: '2026-09-04T13:54:46.187451+00:00'
---
Reset AdaptiveAdam state so a second optimize() is safe; raise on clip_weight under the Venice algebra and guard MMA/OC; expose fix_right_wall on StructuralParams with default True.
