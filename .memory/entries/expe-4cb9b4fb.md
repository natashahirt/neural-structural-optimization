---
id: expe-4cb9b4fb
type: experiment
project: semantopology_hardfork
parent_id: plan-073cebb0
title: Generated wing stencil reorganizes the structural load path
node_label: 'Generated wing stencil reorganizes the structural '
tags: e1,generated-stencil,butterfly-wing,load-path,success
status: active
open_threads: 0
success: 'true'
files: script/resources/results/generated_stencil_butterfly_wing_venation/representative_drawing.png@97ecab1,
  script/resources/results/generated_stencil_butterfly_wing_venation/occupancy.png@97ecab1,
  script/resources/results/generated_stencil_butterfly_wing_venation/summary.json@97ecab1,
  script/resources/results/generated_stencil_butterfly_wing_venation/physics/comparison.png@97ecab1,
  script/resources/results/generated_stencil_butterfly_wing_venation/physics/physical_density.png@97ecab1,
  script/resources/results/generated_stencil_butterfly_wing_venation/physics/summary.json@97ecab1,
  script/clip_dream_layout.py@97ecab1, neural_structural_optimization/models/loss_sketch.py@97ecab1
session_id: sess-0ff8a38f
created_at: '2026-09-06T05:44:27.614533+00:00'
updated_at: '2026-09-06T05:44:27.614533+00:00'
results: '[{"metric": "occupancy_fraction", "value": 0.185, "split": "generated-wing-stencil-prephysics",
  "criterion": "representative sparse stencil", "source": "script/resources/results/generated_stencil_butterfly_wing_venation/summary.json"},
  {"metric": "spatial_mass_loss", "value": 0.7546, "split": "generated-wing-stencil-vs-physics-proxy",
  "criterion": ">=0.25", "source": "script/resources/results/generated_stencil_butterfly_wing_venation/summary.json"},
  {"metric": "top_to_bottom_connected", "value": 1, "unit": "boolean", "split": "generated-wing-stencil-prephysics",
  "criterion": "=1", "source": "script/resources/results/generated_stencil_butterfly_wing_venation/summary.json"},
  {"metric": "compliance", "value": 74.7468, "split": "generated-wing-stencil-physics-final",
  "window": "131 optimization steps", "criterion": "<=85", "source": "script/resources/results/generated_stencil_butterfly_wing_venation/physics/summary.json"},
  {"metric": "mean_physical_density", "value": 0.3, "split": "generated-wing-stencil-physics-final",
  "criterion": "target 0.30", "source": "script/resources/results/generated_stencil_butterfly_wing_venation/physics/summary.json"},
  {"metric": "mass_on_stencil", "value": 0.3558, "split": "generated-wing-stencil-physics-final",
  "source": "script/resources/results/generated_stencil_butterfly_wing_venation/physics/summary.json"},
  {"metric": "spatial_mass_loss", "value": 0.6089, "split": "generated-wing-stencil-physics-final",
  "source": "script/resources/results/generated_stencil_butterfly_wing_venation/physics/summary.json"}]'
---
A text-generated representative butterfly-wing line drawing was cropped, resized to 128x256, thresholded into occupancy, and passed through the proven Stage 6 recipe (occupancy union load sites, init from occupancy, spatial prior 4000->400). Whole-building Venice CLIP stayed active with prompt 'butterfly wing venation'; motif-scale crops were off.

Pre-physics gates: occupancy fraction 0.1850, one connected component, top-to-bottom spanning, and spatial_mass_loss=0.7546 against the physics proxy. The structural run converged after 131 steps at compliance 74.7468 and mean physical density 0.3000. Visual result: physical density preserves a recognizable wing-like global structure with a large curved perimeter, branching diagonal veins/ribs, closed cells, and altered through-building load paths. This is qualitatively different from raw CLIP dreams and motif-scale texture runs.

The final mass_on_stencil is 0.3558 and spatial_mass_loss is 0.6089 because the stencil occupies only 18.5% of the grid and the prior anneals to 400; physics builds adjacent thick members rather than tracing every line exactly. That is compatible with the goal: motif informs load path while physics controls thickness and feasibility.

Conclusion: representative text-generated drawing -> stencil -> Stage 6 prior is already a higher-evidence route than integrating VQGAN into the optimizer. VQGAN remains parked unless this approach fails to generalize to another prompt/prototype.
