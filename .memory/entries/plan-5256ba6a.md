---
id: plan-5256ba6a
type: plan
project: semantopology_hardfork
parent_id: plan-039cabee
title: '3.2 leftover: profile the step loop and take measured speedups'
node_label: '3.2 leftover: profile the step loop and take measu'
tags: stage-2,speed,profile,plan
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-60a28823
created_at: '2026-09-05T05:54:45.352940+00:00'
updated_at: '2026-09-05T05:54:45.352940+00:00'
---
Stage 2 leftovers after the CHOLMOD OMP_NUM_THREADS=1 pin. Already landed: CHOLMOD, OpenMP pin, configure_torch_threads, AdaptivePixel coarse-to-fine, factorization cache size 1 on the adjoint, float64 StructuralLoss cast characterized as not a Venice divergence. Still open: (1) profile a real Venice CLIP Stage 6 step into physics forward / HIPS re-trace backward / CLIP crop+encode / optimizer; (2) raise _get_solver cache above 1 and add a smoke/coarse ExperimentConfig preset; (3) if physics dominates, analytic compliance VJP gated by finite-difference; (4) if CLIP dominates, treat num_augs/encoder/device as config before adding more crop scales. Gate: written per-step breakdown; each kept change has a measured delta and a parity pin.
