---
id: todo-6e2d19ff
type: todo
project: semantopology_hardfork
parent_id: plan-265959fb
title: 'Tests: projection-off compliance unchanged, projection-on volfrac holds, params
  reach physics'
node_label: 'Tests: projection-off compliance unchanged, projec'
tags: stage-1,todo
status: discarded
open_threads: 0
success: 'null'
files: ''
session_id: sess-88c4b9c3
created_at: '2026-09-03T22:10:24.455195+00:00'
updated_at: '2026-09-04T00:27:15.165587+00:00'
---
Three gates. (1) With heavyside=False the reordered chain reproduces pre-restructure compliance. (2) With projection on, mean density still equals volfrac to ~1e-9. (3) filter_width/rmin/beta set in the config demonstrably change physics output. Keep the existing 13 tests green.
