---
id: todo-f81231fb
type: todo
project: semantopology_hardfork
parent_id: deci-ba13459e
title: 'Bounded fix: refuse AdaptiveAdam clip_weight; stop calling volume_actual geometry'
node_label: 'Bounded fix: refuse AdaptiveAdam clip_weight; stop'
tags: stage-3,bounded-fix,clip_weight,volume_actual
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-a469ebb9
created_at: '2026-09-04T17:55:12.535588+00:00'
updated_at: '2026-09-04T17:55:12.535588+00:00'
---
Bounded Stage 3 follow-up after the critic FAIL and a successful visual replay: (1) AdaptiveAdam must refuse clip_weight under the Venice preset, the same silent-swallow already refused on get_total_loss; (2) volume_actual stays as a fill-fraction pin and must not be advertised as geometry. Do not chase a tighter correlation floor; the side-by-side image is the geometric comparison.
