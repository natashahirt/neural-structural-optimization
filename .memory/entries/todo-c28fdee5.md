---
id: todo-c28fdee5
type: todo
project: semantopology_hardfork
parent_id: plan-265959fb
title: Reorder physical_density to sigmoid/mask/filter/projection/volume-enforcement
  with a new find_root bracket
node_label: Reorder physical_density to sigmoid/mask/filter/pr
tags: stage-1,todo
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-88c4b9c3
created_at: '2026-09-03T22:10:18.027878+00:00'
updated_at: '2026-09-03T22:10:18.027878+00:00'
---
Move volume enforcement to the end of the chain and add opt-in Heaviside projection before it. Requires a new residual and new bisection bounds: the existing bracket logit(average)-max(x) to logit(average)-min(x) is derived for the plain sigmoid and bisection with a bad bracket silently converges to a bound.
