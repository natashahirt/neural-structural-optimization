---
id: expe-2293aefa
type: experiment
project: semantopology_hardfork
parent_id: plan-3d516df5
title: 'No-occupancy motif-scale look: gothic frame, two-size gate still open'
node_label: 'No-occupancy motif-scale look: gothic frame, two-s'
tags: clip-scale,no-occupancy,experiment,not-silhouette
status: active
open_threads: 0
success: 'false'
files: script/venice_golden_250214.py@739db61, script/resources/results/clip_motif_scale_no_occupancy/summary.json@739db61,
  script/resources/results/clip_motif_scale_no_occupancy/comparison.png@739db61, script/resources/results/clip_motif_scale_no_occupancy/sketch_run.png@739db61
session_id: sess-ee90841b
created_at: '2026-09-05T14:37:14.038742+00:00'
updated_at: '2026-09-05T19:08:20.239543+00:00'
results: '[{"metric": "compliance", "value": 80.04190063476562, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "clip_loss", "value": 598.6527709960938, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.747924268245697, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "volume_actual", "value": 0.322021484375, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "clip_motif_f1p0000", "value": 0.30972903966903687, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "clip_motif_f0p2500", "value": 0.3577222526073456, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "clip_motif_f0p0625", "value": 0.39426785707473755, "split": "venice-250214-motif-scale-no-occupancy",
  "window": "final-step", "source": "script/resources/results/clip_motif_scale_no_occupancy/summary.json"},
  {"metric": "compliance", "value": 73.99691670938864, "split": "venice-250214-GOLDEN_FINAL",
  "window": "final-step", "source": "script/venice_golden_250214.py@739db61"}]'
related_to: expe-79957d80,expe-083c1ce7
---
Venice 250214 skeletons + physical-scale CLIP crops, no sketch occupancy and no spatial anneal. Seed is the golden image. Fracs (1.0, 0.25, 0.0625), 4 crops/scale, weight 1.0. Wrote script/resources/results/clip_motif_scale_no_occupancy/.

Vs Venice RRC reference (GOLDEN_FINAL in script/venice_golden_250214.py): compliance 73.997→80.042; clip_loss 283.37→598.65 (extra term is inside the same scalar; clip_loss_raw 0.383→0.748); volume_actual 0.315→0.322. Per-scale crop distances at the last step: building 0.310, storey 0.358, member 0.394 (mean 0.354) — same band as the occupancy+motif look (expe-79957d80).

Occupancy isolation worked: the look is the gothic frame, not sketch 12's V. Motifs (jaw/teeth on a span, skull/ribs) are more literal than the occupancy run. Two-size gate still not clearly met — one bay-scale skeleton language on the Venice silhouette, not the same language at storey vs member length.
