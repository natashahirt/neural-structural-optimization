---
id: expe-a55324ba
type: experiment
project: semantopology_hardfork
parent_id: expe-fc0f2350
title: Sculptural filter-then-project view cuts CLIP's low-density ink without costing
  compliance
node_label: Sculptural filter-then-project view cuts CLIP's lo
tags: clip,semantic-prior,projection,heaviside,minimum-length-scale,ink-fraction,skull
status: active
open_threads: 0
success: 'true'
files: neural_structural_optimization/models/loss_semantic_prior.py@8f1dafb, neural_structural_optimization/tests/test_semantic_prior.py@8f1dafb,
  script/clip_saliency_compliance.py@8f1dafb, script/resources/results/clip_saliency_compliance_human_skull_projected/summary.json@8f1dafb,
  script/resources/results/clip_saliency_compliance_human_skull_hierarchical/summary.json@8f1dafb
session_id: sess-655cccb4
created_at: '2026-09-06T19:46:48.896663+00:00'
updated_at: '2026-09-06T19:46:56.673341+00:00'
results: '[{"metric": "preference_ink_fraction", "value": 0.4028, "unit": "fraction",
  "split": "human_skull_projected (beta=8, sigma=2, global)", "window": "final step
  of 40", "criterion": "below the beta=0 baseline 0.6824", "source": "clip_saliency_compliance_human_skull_projected"},
  {"metric": "preference_ink_fraction", "value": 0.6824, "unit": "fraction", "split":
  "human_skull_hierarchical (beta=0 baseline)", "window": "final step of 40", "source":
  "clip_saliency_compliance_human_skull_hierarchical"}, {"metric": "clip_compliance_overlap",
  "value": 0.1934, "unit": "fraction", "split": "human_skull_projected (beta=8, sigma=2,
  global)", "window": "final step of 40", "source": "clip_saliency_compliance_human_skull_projected"},
  {"metric": "clip_compliance_overlap", "value": 0.1436, "unit": "fraction", "split":
  "human_skull_hierarchical (beta=0 baseline)", "window": "final step of 40", "source":
  "clip_saliency_compliance_human_skull_hierarchical"}, {"metric": "compliance", "value":
  186.2992, "unit": "compliance", "split": "human_skull_projected (beta=8, sigma=2,
  global)", "window": "final step of 40", "criterion": "no worse than the beta=0 baseline
  187.14", "source": "clip_saliency_compliance_human_skull_projected"}, {"metric":
  "compliance", "value": 187.1445, "unit": "compliance", "split": "human_skull_hierarchical
  (beta=0 baseline)", "window": "final step of 40", "source": "clip_saliency_compliance_human_skull_hierarchical"},
  {"metric": "component_count", "value": 1, "unit": "components", "split": "human_skull_projected
  (beta=8, sigma=2, global)", "window": "final step of 40", "source": "clip_saliency_compliance_human_skull_projected"},
  {"metric": "component_count", "value": 7, "unit": "components", "split": "human_skull_hierarchical
  (beta=0 baseline)", "window": "final step of 40", "source": "clip_saliency_compliance_human_skull_hierarchical"}]'
related_to: note-352e8ecf
---
Closes the gray-drawing loophole from note-352e8ecf. Scales listed in `projection_scales` are now scored on a filter-then-project view of physical density (Gaussian blur for a minimum feature size, then a tanh Heaviside projection at eta=0.5), while physics keeps seeing the unprojected field. Projection is confined to the sculptural scale by default (`projection_scales=('global',)`), so the storey and member stages still score raw density and can resolve fine detail. `projection_beta=0` restores the previous behaviour exactly and the full suite stays green (296 passed, 8 skipped).

Matched arms, both hierarchical curriculum, 40 steps, seed 12, volfrac 0.3, 32x64 coarse grid. The projected arm used beta=8, filter sigma=2 cells, global scale only.

| metric | hierarchical (beta=0) | projected (beta=8) |
|---|---|---|
| compliance | 187.14 | 186.30 |
| clip_loss_raw | 0.41172 | 0.41191 |
| preference ink fraction | 0.6824 | 0.4028 |
| CLIP-compliance overlap | 0.1436 | 0.1934 |
| gradient cosine | -0.0494 | -0.0727 |
| component count | 7 | 1 |
| spanning mass fraction | 0.7466 | 0.7900 |

Reading: the semantic score and the compliance are both effectively unchanged, but the preference driving them moved off near-void cells and onto load-bearing material, and the structure collapsed from seven components to one. The motif is being paid for in structure rather than ink at no measured physics cost.

Caveats. This is a single seed and a single prompt, so treat the magnitudes as indicative. Gradient cosine did not improve and in fact got slightly more negative, so CLIP and compliance still disagree on direction; only the spatial overlap improved. The oversized auto-ratio weight is still present (1567 here, 1922 in the baseline) and remains unfixed. The new `preference_ink_fraction` metric is the standing instrument for this failure mode and should be reported on every future prior run.
