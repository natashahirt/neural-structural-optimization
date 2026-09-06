---
id: expe-54faec24
type: experiment
project: semantopology_hardfork
parent_id: expe-a55324ba
title: 'Projected full-res skull replay: motif returns as gray ink at detail scales'
node_label: 'Projected full-res skull replay: motif returns as '
tags: clip,semantic-prior,projection,full-replay,skull,venation,low-density-ink
status: active
open_threads: 0
success: 'false'
files: script/resources/results/clip_saliency_compliance/human_skull_projected_full/summary.json@2455bfa,
  script/resources/results/clip_saliency_compliance/human_skull_projected_full/comparison.png@2455bfa,
  script/resources/results/clip_saliency_compliance/human_skull_projected_full/physical_density.png@2455bfa,
  script/resources/results/clip_saliency_compliance/butterfly_wing_venation_projected/summary.json@2455bfa
session_id: sess-655cccb4
created_at: '2026-09-06T20:08:13.321280+00:00'
updated_at: '2026-09-06T20:08:13.321280+00:00'
results: '[{"metric": "compliance", "value": 93.0694, "unit": "compliance", "split":
  "human_skull_projected_full (beta=8, sigma=2, global)", "window": "final step of
  80 at 128x256", "criterion": "compare to unprojected full replay 80.46", "source":
  "clip_saliency_compliance/human_skull_projected_full"}, {"metric": "clip_loss_raw",
  "value": 0.3989, "unit": "distance", "split": "human_skull_projected_full (beta=8,
  sigma=2, global)", "window": "final step of 80 at 128x256", "source": "clip_saliency_compliance/human_skull_projected_full"},
  {"metric": "preference_ink_fraction", "value": 0.4829, "unit": "fraction", "split":
  "human_skull_projected_full (beta=8, sigma=2, global)", "window": "final step of
  80 at 128x256", "source": "clip_saliency_compliance/human_skull_projected_full"},
  {"metric": "clip_compliance_overlap", "value": 0.1781, "unit": "fraction", "split":
  "human_skull_projected_full (beta=8, sigma=2, global)", "window": "final step of
  80 at 128x256", "source": "clip_saliency_compliance/human_skull_projected_full"},
  {"metric": "compliance", "value": 199.3965, "unit": "compliance", "split": "butterfly_wing_venation_projected
  (beta=8, sigma=2, hierarchical)", "window": "final step of 40 at 32x64", "source":
  "clip_saliency_compliance/butterfly_wing_venation_projected"}]'
---
128×256 hierarchical replay of the sculptural-projection prior (beta=8, sigma=2, global scale only; storey/member still score raw density). Same seed 12, 80 steps, volfrac 0.3, as the unprojected full replay in note-352e8ecf.

| metric | unprojected full replay | projected full |
|---|---|---|
| compliance | 80.46 | 93.07 |
| clip_loss_raw | 0.4027 | 0.3989 |
| CLIP-compliance overlap | 0.1106 | 0.1781 |
| gradient cosine | -0.0439 | -0.0728 |
| preference ink fraction | (not logged) | 0.4829 |
| spanning mass | 0.9019 | 0.8720 |
| components | 15 | 19 |
| connected | yes | yes |

Reading: projection did not turn the storey-scale skull into a load path. The density field is still thick black columns with faint gray stacked skulls in the bays — the same ink morphology as the unprojected replay, slightly less cheap (overlap up, CLIP a hair better, compliance ~16% worse). Ink fraction 0.48 at full res is higher than the coarse projected arm (0.40) because the later unprojected storey/member stages are allowed to draw with gray, which is the intended split. Closing ink at the sculptural scale does not prevent CLIP from hatching the motif once detail scales unlock.

Venation projected (coarse, hierarchical, same projection) also finished: compliance 199.40, clip_loss_raw 0.4071, ink 0.438, connected, 8 components. Not matched to the earlier global-only venation prior.
