---
id: expe-fc0f2350
type: experiment
project: semantopology_hardfork
parent_id: plan-d918345f
title: Coarse skull CLIP/SDS spatial-prior arms
node_label: Coarse skull CLIP/SDS spatial-prior arms
tags: clip,sds,spatial-prior,experiment,skull,venation
status: active
open_threads: 3
success: 'true'
files: script/clip_saliency_compliance.py@ceb3f48, neural_structural_optimization/models/loss_semantic_prior.py@ceb3f48,
  script/resources/results/clip_saliency_compliance_index/summary.json@c94f3aa
session_id: sess-5c3dbfc6
created_at: '2026-09-06T18:42:16.508477+00:00'
updated_at: '2026-09-08T17:45:45.971280+00:00'
results: '[{"metric": "compliance", "value": 202.0467529296875, "split": "32x64-four-storey-40step",
  "window": "human-skull-compliance-only", "source": "clip_saliency_compliance_human_skull_compliance/summary.json"},
  {"metric": "compliance", "value": 238.31710815429688, "split": "32x64-four-storey-40step",
  "window": "human-skull-scalar-clip", "source": "clip_saliency_compliance_human_skull_scalar_clip/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.4292440116405487, "split": "32x64-four-storey-40step",
  "window": "human-skull-scalar-clip", "source": "clip_saliency_compliance_human_skull_scalar_clip/summary.json"},
  {"metric": "compliance", "value": 249.63784790039062, "split": "32x64-four-storey-40step",
  "window": "human-skull-clip-prior-global", "source": "clip_saliency_compliance_human_skull_clip_prior/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.4141008257865906, "split": "32x64-four-storey-40step",
  "window": "human-skull-clip-prior-global", "source": "clip_saliency_compliance_human_skull_clip_prior/summary.json"},
  {"metric": "grad_cosine", "value": -0.08096852153539658, "split": "32x64-four-storey-40step",
  "window": "human-skull-clip-prior-global", "source": "clip_saliency_compliance_human_skull_clip_prior/summary.json"},
  {"metric": "compliance", "value": 187.14451599121094, "split": "32x64-four-storey-40step",
  "window": "human-skull-clip-prior-hierarchical", "source": "clip_saliency_compliance_human_skull_hierarchical/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.4117206037044525, "split": "32x64-four-storey-40step",
  "window": "human-skull-clip-prior-hierarchical", "source": "clip_saliency_compliance_human_skull_hierarchical/summary.json"},
  {"metric": "compliance", "value": 231.6792755126953, "split": "32x64-four-storey-40step",
  "window": "butterfly-wing-venation-clip-prior-global", "source": "clip_saliency_compliance_butterfly_wing_venation_clip_prior/summary.json"},
  {"metric": "compliance", "value": 252.14523315429688, "split": "32x64-four-storey-40step",
  "window": "human-skull-sds-prior-global", "source": "clip_saliency_compliance_human_skull_sds_prior/summary.json"},
  {"metric": "compliance", "value": 80.45938110351562, "split": "128x256-80step-hierarchical-replay",
  "window": "human-skull-full-replay", "criterion": "<=85", "source": "clip_saliency_compliance_human_skull_full_replay/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.4026639759540558, "split": "128x256-80step-hierarchical-replay",
  "window": "human-skull-full-replay", "source": "clip_saliency_compliance_human_skull_full_replay/summary.json"}]'
related_to: expe-d7945d40
---
Matched 32x64 / 40-step arms on `human skull` (seed 12, volfrac 0.3, 4 storeys). Live prior is relu(-dL/d rho) EMA occupancy plus Stage-6 mass loss; FEA compliance is unmodified.

- compliance-only: C=202.05, top-to-bottom connected
- scalar CLIP: C=238.32, clip_loss_raw=0.4292
- CLIP prior (global): C=249.64, clip_loss_raw=0.4141 (better semantic than scalar), connected, grad_cosine=-0.081, overlap=0.175. Auto-ratio 0.25 produced weight=1718 because ||g_prior|| << ||g_C||.
- hierarchical CLIP: C=187.14 (best physics of the CLIP arms), clip_loss_raw=0.4117, storey map on after first upsample
- SDS frozen-tiny-UNet prior: C=252.15, clip_loss_raw=0.4238, lost top-to-bottom connectivity. This is the SDS interface, not a pretrained image model.
- venation CLIP prior: C=231.68, clip_loss_raw=0.4045, connected
- full 128x256 hierarchical replay (gated: connected and better CLIP than scalar): C=80.46, clip_loss_raw=0.4027, 15 components, spanning mass 0.90. Skull-like motifs appear at member scale along load paths.

CLIP and compliance gradients stay weakly anti-aligned (~-0.05 to -0.08). The spatial prior moves CLIP a little without collapsing physics; it does not yet make a coarse skull the load path.
