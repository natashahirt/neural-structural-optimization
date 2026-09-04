---
id: note-7dabd6e1
type: note
project: semantopology_hardfork
parent_id: plan-e9993b76
title: 'BLOCKER for CNN+pixel residual stage: constrained_logits double-projects,
  +-8-11% volume violation with beta-dependent sign'
node_label: 'BLOCKER for CNN+pixel residual stage: constrained_'
tags: blocker,stage-8,cnn-pixel-residual,projection,deferred
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/train/utils.py@d6106bb
session_id: sess-bfd66d57
created_at: '2026-09-04T00:06:13.645758+00:00'
updated_at: '2026-09-04T18:32:58.355934+00:00'
results: '[{"metric": "handoff_volume_error", "value": -7.5, "unit": "%", "split":
  "multistory_building volfrac=0.3 beta=4", "criterion": "should equal volfrac", "source":
  "stage-1-critic-pass-2"}, {"metric": "handoff_volume_error", "value": 8.5, "unit":
  "%", "split": "multistory_building volfrac=0.3 beta=16", "criterion": "should equal
  volfrac", "source": "stage-1-critic-pass-2"}, {"metric": "handoff_volume_error",
  "value": -6.4, "unit": "%", "split": "crane volfrac=0.3 beta=4", "criterion": "should
  equal volfrac", "source": "stage-1-critic-pass-2"}, {"metric": "handoff_volume_error",
  "value": 11.2, "unit": "%", "split": "crane volfrac=0.3 beta=16", "criterion": "should
  equal volfrac", "source": "stage-1-critic-pass-2"}]'
related_to: expe-38a0e2f4
---
Found by adversarial review during stage 1. Deferred here by user decision rather than fixed, because it only bites when Heaviside projection is enabled (opt-in, off by default) and the handoff is redesigned in this stage anyway.

THE DEFECT: with heavyside=True, train/utils.py constrained_logits hands the pixel model a density that has ALREADY been projected, and the pixel objective projects it AGAIN. Nothing inside the objective restores the volume, because the pixel path evaluates volume_constraint=False; only MMA's separate inequality constrains it, and OC's find_root starts roughly 10% off.

MEASURED, multistory_building at volfrac 0.3: the handoff field has volume 0.35959 (beta=4) and 0.41741 (beta=16); after the pixel objective re-filters and re-projects it lands at 0.27750 (-7.5%) and 0.32553 (+8.5%). crane: 0.28082 (-6.4%) and 0.33348 (+11.2%).

WHY IT BLOCKS THE RESIDUAL WORK: the sign of the violation FLIPS between beta=4 and beta=16. So a beta continuation schedule, which is the natural thing to want for CNN-to-pixel transfer, makes the handoff volume oscillate rather than converge. The 'pixel locks in the structure quickly' intuition depends on the handoff being faithful, and under projection it is not.

RELATED ROOT CAUSE: this is the same underlying problem as the deferred stage-4 item, namely that the objective and the render disagree about what the physical density IS. Fixing the canonical density in stage 4 may resolve or reshape this; re-measure before designing around it.
