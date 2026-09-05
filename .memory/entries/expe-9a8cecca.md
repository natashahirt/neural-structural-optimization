---
id: expe-9a8cecca
type: experiment
project: semantopology_hardfork
parent_id: plan-5256ba6a
title: 'Stage 2 profile: CLIP owns the 128x256 Venice step'
node_label: 'Stage 2 profile: CLIP owns the 128x256 Venice step'
tags: stage-2,profile,clip-bound,experiment
status: active
open_threads: 0
success: 'true'
files: script/profile_step.py@1226072, script/resources/results/stage2_profile_step.json@1226072,
  script/resources/results/stage2_profile_step_full.json@1226072, neural_structural_optimization/structural/autograd.py@1226072,
  neural_structural_optimization/experiment.py@9ab1887
session_id: sess-60a28823
created_at: '2026-09-05T06:18:45.150784+00:00'
updated_at: '2026-09-05T06:18:45.150784+00:00'
results: '[{"metric": "combined_step", "value": 8.662947248027194, "unit": "s", "split":
  "venice-250214-128x256-32augs-cpu", "window": "n=1", "source": "script/resources/results/stage2_profile_step_full.json@1226072"},
  {"metric": "clip_share", "value": 7.31236366200028, "unit": "s", "split": "venice-250214-128x256-32augs-cpu",
  "window": "fwd+isolated-bwd", "source": "script/resources/results/stage2_profile_step_full.json@1226072"},
  {"metric": "physics_share", "value": 0.6525533410022035, "unit": "s", "split": "venice-250214-128x256-32augs-cpu",
  "window": "fwd+isolated-bwd", "source": "script/resources/results/stage2_profile_step_full.json@1226072"},
  {"metric": "clip_forward", "value": 2.66134198801592, "unit": "s", "split": "venice-250214-128x256-32augs-cpu",
  "window": "n=1", "source": "script/resources/results/stage2_profile_step_full.json@1226072"},
  {"metric": "combined_step", "value": 0.894303282975064, "unit": "s", "split": "venice-250214-smoke-8x16-4augs-cpu",
  "window": "n=3", "source": "script/resources/results/stage2_profile_step.json@1226072"}]'
---
Profiled a real Venice CLIP step on this machine. Smoke (8x16, 4 augs, n=3): combined step 0.894s; CLIP fwd+isolated-bwd 0.979s; physics fwd+isolated-bwd 0.018s. Full 128x256, 32 augs, CPU, ViT-B/32, n=1: combined step 8.663s; CLIP share 7.312s (fwd 2.661s, isolated bwd 4.651s); physics share 0.653s (fwd 0.457s, isolated bwd 0.195s). CLIP owns the step (~11x physics at the Stage 6 grid). Isolated backward overcounts vs the combined backward (4.079s) because it rebuilds a graph; the ranking is unchanged.

Kept: SOLVER_CACHE_MAXSIZE=8 (metrics eval cannot evict the factorization); `--smoke` / venice_250214_smoke; num_augs, encoder, and device already live on ClipConfig / RunConfig. Not started: analytic compliance VJP. Next CLIP-speed play is fewer augs or a GPU encoder, before stacking more crop scales on the default path. venice_250214_motif_scale is opt-in.
