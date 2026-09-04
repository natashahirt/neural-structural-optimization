---
id: chec-4f90e40c
type: checkpoint
project: semantopology_hardfork
parent_id: plan-c9c6884b
title: 'Stage 5 complete: typed ExperimentConfig and venice_250214 preset'
node_label: 'Stage 5 complete: typed ExperimentConfig and venic'
tags: stage-5,checkpoint,experiment-config
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/experiment.py@54957c4, neural_structural_optimization/__init__.py@54957c4,
  neural_structural_optimization/tests/test_experiment.py@54957c4, script/venice_golden_250214.py@54957c4,
  script/run.py@54957c4
session_id: sess-a469ebb9
created_at: '2026-09-04T19:13:05.641176+00:00'
updated_at: '2026-09-04T19:13:05.641176+00:00'
results: '[{"metric": "pytest_passed", "value": 177, "unit": "1", "split": "neural_structural_optimization/tests",
  "window": "no-VENICE_PARITY_FULL", "criterion": "green after Stage 5", "source":
  "pytest-stage-5"}, {"metric": "pytest_skipped", "value": 8, "unit": "1", "split":
  "neural_structural_optimization/tests", "window": "no-VENICE_PARITY_FULL", "criterion":
  "full golden replay remains gated", "source": "pytest-stage-5"}]'
---
Stage 5 slice landed: frozen ExperimentConfig with sections (problem, model, clip, optimizer, init, run), resolve(), with_overrides, JSON round-trip, presets venice_250214 and venice_250214_smoke. VeniceGoldenConfig / GOLDEN / SMOKE live in experiment.py so --print-config does not import the golden script's CLIP builders; the golden script re-exports them. Gate: venice_250214().to_venice_golden() == GOLDEN. CLI: python -m neural_structural_optimization.experiment --print-config venice_250214 (no clip/kornia). script/run.py dispatches dashed argv to that CLI before loading CLIPLoss; no-flag invocation is still the CNN demo. --run delegates to the existing golden builders. CLIP_AVAILABLE is lazy; torch still imports at package init so CHOLMOD does not race a late OpenMP load.

OUT held: sweeps, Stage 6/7 knobs, loss algebra, TF32, CNN residual, constrained_logits.

Suite: 177 passed, 8 skipped (VENICE_PARITY_FULL gated).
