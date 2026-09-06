---
id: expe-b544bb5e
type: experiment
project: semantopology_hardfork
parent_id: deci-cdd1e688
title: 'Neutral-init motif-scale look: three-bay layout, CLIP motifs survive'
node_label: 'Neutral-init motif-scale look: three-bay layout, C'
tags: clip-scale,neutral-init,no-occupancy,experiment
status: active
open_threads: 1
success: 'true'
files: neural_structural_optimization/train/utils.py@3a308e6, neural_structural_optimization/experiment.py@3a308e6,
  script/venice_golden_250214.py@3a308e6, neural_structural_optimization/tests/test_neutral_init.py@3a308e6,
  script/resources/results/clip_motif_scale_neutral_init/summary.json@3a308e6, script/resources/results/clip_motif_scale_neutral_init/comparison.png@3a308e6,
  script/resources/results/clip_motif_scale_neutral_init/sketch_run.png@3a308e6
session_id: sess-c5f62176
created_at: '2026-09-06T02:46:20.670105+00:00'
updated_at: '2026-09-06T03:24:08.198639+00:00'
has_child_rationale: Tests the decision that dropping the Venice image seed lets CLIP
  move coarse topology.
results: '[{"metric": "compliance", "value": 71.77558898925781, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.739435076713562, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "clip_loss", "value": 530.73388671875, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "volume_actual", "value": 0.319580078125, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "clip_motif_f1p0000", "value": 0.32656627893447876, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "clip_motif_f0p2500", "value": 0.3608534634113312, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "clip_motif_f0p0625", "value": 0.3758782744407654, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "clip_motif_mean", "value": 0.3544326722621918, "split": "clip_motif_scale_neutral_init",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"},
  {"metric": "steps", "value": 135, "split": "clip_motif_scale_neutral_init", "window":
  "full-run", "source": "script/resources/results/clip_motif_scale_neutral_init/summary.json"}]'
related_to: expe-2293aefa,note-2a958be2
---
Venice 250214 skeletons + physical-scale CLIP crops, no occupancy, **neutral init** (uniform volfrac + noise_amp=0.01, union_load_sites=true). Fracs (1.0, 0.25, 0.0625), 4 crops/scale, weight 1.0. Wrote `script/resources/results/clip_motif_scale_neutral_init/` — does not overwrite the image-seeded teacher at `clip_motif_scale_no_occupancy/`.

Vs image-seeded teacher (expe-2293aefa, split venice-250214-motif-scale-no-occupancy, window final-step): compliance 80.042→71.776; clip_loss_raw 0.748→0.739; volume_actual 0.322→0.320; clip_motif_mean 0.354→0.354. Converged in 135 steps (resize 48, 49) vs 121 (resize 48, 69). First-step compliance was 604.96 at 32x64, vs ~150 on the image-seeded run — the frame seed is gone.

Visual vs image-seeded prior: layout **changed**. Neutral init grew three evenly spaced vertical bays with gothic arches at the top, instead of Venice's two-pillar / one large arch silhouette. Skulls, ribcages, and grasping hands are still readable in the members. That is the lock-in hypothesis from deci-cdd1e688: CLIP can move coarse topology when it is not handed a finished gothic frame.

The two-size gate of plan-3d516df5 is still not closed — this run tests initialization, not whether skeleton language repeats at storey vs member length. Frozen venice_250214 GOLDEN remains image-seeded.
