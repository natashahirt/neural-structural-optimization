---
id: deci-cdd1e688
type: decision
project: semantopology_hardfork
parent_id: plan-3d516df5
title: Neutral initialization is the default for new motif runs
node_label: Neutral initialization is the default for new moti
tags: initialization,neutral-init,clip-scale,motif-layout,default
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/experiment.py@c03246a, script/venice_golden_250214.py@9816779
session_id: sess-82329cef
created_at: '2026-09-06T01:57:30.041939+00:00'
updated_at: '2026-09-06T01:57:30.041939+00:00'
has_child_rationale: Allow CLIP to influence coarse topology before a frame is locked.
---
User chose neutral initialization as the default for new motif-guided experiments. Starting from Venice thick_outer_lins.png pre-commits the architectural frame before CLIP can influence topology. New motif runs should start from uniform volfrac plus small deterministic noise, with only physically mandatory load/support collectors added if needed. The frozen venice_250214 compatibility preset must retain its historical image seed for bit-exact reproduction; neutral becomes the research default, not a retroactive change to GOLDEN.
