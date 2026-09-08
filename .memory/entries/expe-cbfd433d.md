---
id: expe-cbfd433d
type: experiment
project: semantopology_hardfork
parent_id: plan-44c213c9
title: In-loop binary projection does not preserve a useful skull silhouette
node_label: In-loop binary projection does not preserve a usef
tags: ''
status: active
open_threads: 0
success: 'false'
files: script/clip_dream_layout.py@a91aa20
session_id: sess-85fd29d6
created_at: '2026-09-08T20:42:37.275843+00:00'
updated_at: '2026-09-08T20:42:37.275843+00:00'
results: '[{"metric": "neighbor_contrast_ratio", "value": 0.03176801545954727, "unit":
  "ratio", "split": "48x24-control sigma-2.0", "window": "300 dream steps", "criterion":
  "visually useful skull silhouette", "source": "/tmp/skull_proj_v040/summary.json"},
  {"metric": "component_count", "value": 1, "unit": "components", "split": "48x24-control
  sigma-2.0", "window": "final projected field", "criterion": "visually useful skull
  silhouette", "source": "/tmp/skull_proj_v040/summary.json"}, {"metric": "neighbor_contrast_ratio",
  "value": 0.04794608320445168, "unit": "ratio", "split": "48x24-control sigma-0.8",
  "window": "150 dream steps", "criterion": "visually useful skull silhouette", "source":
  "/tmp/skull_proj_s08/summary.json"}, {"metric": "component_count", "value": 3, "unit":
  "components", "split": "48x24-control sigma-0.8", "window": "final projected field",
  "criterion": "visually useful skull silhouette", "source": "/tmp/skull_proj_s08/summary.json"}]'
---
Compared otherwise matched 48x24-control CLIP-only human-skull dreams at target projected mean 0.4 and beta annealed from 1 to 8. Sigma 2.0 collapsed to one smooth bottle-like blob. Reducing sigma to 0.8 recovered eye-like holes and greater edge contrast, but yielded a fragmented three-component form with a detached lower mass; neither output is a useful silhouette. This rules out blur strength as the sole cause. Do not continue tuning sigma alone; preserve CLIP's continuous grayscale representation and test it as a soft physics prior, or derive a differentiable edge/interior representation without forcing CLIP itself to solve the binary topology.
