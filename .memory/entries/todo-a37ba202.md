---
id: todo-a37ba202
type: todo
project: semantopology_hardfork
parent_id: plan-3d516df5
title: 'E2: CNN parameterization, CLIP held fixed'
node_label: 'E2: CNN parameterization, CLIP held fixed'
tags: clip-scale,e2,cnn,parameterization,parked,todo
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-c5f62176
created_at: '2026-09-06T04:04:59.124512+00:00'
updated_at: '2026-09-06T04:04:59.124512+00:00'
---
Parked. Isolate parameterization vs CLIP signal: hold venation motif-scale CLIP and Venice algebra fixed, replace AdaptivePixel with CNNModel (Hoyer reparameterization: 128-dim latent → dense → conv upsample to 128×256). No resolution ladder — the CNN *is* the multi-scale prior. Calibrate lr with a structure-only CNN run first (Venice lr=0.2 does not transfer). Then CLIP run vs pixel venation look. Add a frozen-decoder (latent-only) variant: jointly trained conv weights can still paint texture. Draft runner already started at script/cnn_motif_scale.py — do not run until E1 is judged.

This tests member *language*, not load path. Do not start until E1 reports.
