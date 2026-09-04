---
id: plan-c9c6884b
type: plan
project: semantopology_hardfork
parent_id: plan-e9993b76
title: '3.4. Stage 5: Typed experiment config'
node_label: '3.4. Stage 5: Typed experiment config'
tags: stage-5,experiment-config,plan
status: active
open_threads: 2
success: 'null'
files: ''
session_id: sess-a469ebb9
created_at: '2026-09-04T18:44:46.620083+00:00'
updated_at: '2026-09-04T19:54:43.170053+00:00'
has_child_rationale: Stage 5 is the next gate after canonical density; sequencing
  decision deci-640e26f0 parks CNN handoff until Stage 8.
related_to: deci-640e26f0,deci-6d4f7d88
---
GOAL: typed experiment interface so Stages 6-7 are comparable runs, not hardcoded script/run.py.

IN: frozen ExperimentConfig wrapping StructuralParams, model, optimizer, CLIP, Venice preset flag, seed/device/output. resolve(), with_overrides, JSON round-trip. venice_250214 preset must match VeniceGoldenConfig knobs. Thin runner. --print-config that does not load CLIP.

OUT: sweeps, Stage 6/7 objective knobs, changing loss algebra, flipping TF32 globally, CNN+pixel residual fields, constrained_logits / handoff (Stage 8).

GATE: venice_250214 resolves to the same knobs the golden script already uses. No new quality claim.
