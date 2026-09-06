---
id: note-352e8ecf
type: note
project: semantopology_hardfork
parent_id: expe-fc0f2350
title: Large CLIP skull is likely low-density semantic ink
node_label: Large CLIP skull is likely low-density semantic in
tags: clip,semantic-prior,skull,low-density-ink,structural-participation
status: active
open_threads: 0
success: 'null'
files: script/resources/results/clip_saliency_compliance_human_skull_full_replay/summary.json@5df9240,
  script/resources/results/clip_saliency_compliance_human_skull_full_replay/physical_density.png@5df9240,
  script/resources/results/clip_saliency_compliance_human_skull_full_replay/comparison.png@5df9240
session_id: sess-655cccb4
created_at: '2026-09-06T19:19:30.090577+00:00'
updated_at: '2026-09-06T19:46:56.673341+00:00'
results: '[{"metric": "clip_compliance_overlap", "value": 0.110626220703125, "unit":
  "fraction", "split": "human_skull_full_replay", "window": "final step", "source":
  "script/resources/results/clip_saliency_compliance_human_skull_full_replay/summary.json"},
  {"metric": "clip_compliance_gradient_cosine", "value": -0.04391924664378166, "unit":
  "cosine", "split": "human_skull_full_replay", "window": "final step", "source":
  "script/resources/results/clip_saliency_compliance_human_skull_full_replay/summary.json"}]'
related_to: expe-a55324ba
---
The recognizable storey-scale skull in the 128×256 hierarchical replay is visually carried mainly by pale gray density, while the near-black columns and braces remain the dominant spanning load paths. Treat this as a hypothesis strongly supported by the diagnostics, not yet a causal proof: the final CLIP–compliance overlap was 0.110626 and the gradient cosine was −0.043919. The next experiment should test structural participation directly with density projection/thresholding, strain-energy localization, and equal-mass semantic-region versus random-region ablations. The design target is to prevent CLIP from satisfying the prompt using mechanically negligible gray material.
