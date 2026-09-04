---
id: anno-c9f272ea
type: annotation
project: semantopology_hardfork
parent_id: deci-ba13459e
title: Always save a visual point of comparison for design runs
node_label: Always save a visual point of comparison for desig
tags: preference,visualization,parity
status: active
open_threads: 0
success: 'null'
files: script/venice_golden_250214.py@139a2a4
session_id: sess-d648f5b1
created_at: '2026-09-04T15:32:50.472001+00:00'
updated_at: '2026-09-04T15:32:50.472001+00:00'
---
Whenever a run, replay, or parity check produces a design, always write a visual: the design image itself plus a side-by-side against the relevant reference when one exists. Scalar logs and correlation numbers are not a substitute. This is a standing preference, not a one-off for the 250214 golden replay.

The first place this is wired is `script/venice_golden_250214.py`, which now writes `venice_golden_250214_replay.png` and `venice_golden_250214_comparison.png` next to the trajectory JSON.
