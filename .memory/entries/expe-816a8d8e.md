---
id: expe-816a8d8e
type: experiment
project: semantopology_hardfork
parent_id: plan-cbdb1948
title: Storey/member scaffold student look vs raw motif teacher
node_label: Storey/member scaffold student look vs raw motif t
tags: clip-scale,motif-layout,self-distillation,skeletons,experiment
status: active
open_threads: 0
success: 'true'
files: neural_structural_optimization/models/loss_sketch.py@2c13e72, neural_structural_optimization/experiment.py@c03246a,
  script/motif_layout_distillation.py@2c13e72, neural_structural_optimization/tests/test_sketch.py@2c13e72,
  neural_structural_optimization/tests/test_experiment.py@2c13e72, neural_structural_optimization/tests/test_clip_motif_scale.py@2c13e72,
  script/resources/results/clip_motif_layout_no_occupancy/summary.json@2c13e72, script/resources/results/clip_motif_layout_no_occupancy/comparison.png@2c13e72
session_id: sess-4872d00d
created_at: '2026-09-05T22:34:16.969867+00:00'
updated_at: '2026-09-05T22:34:25.510742+00:00'
has_child_rationale: Completed teacher/student layout look against the raw motif teacher
  baseline.
results: '[{"metric": "compliance", "value": 76.96536254882812, "split": "student-storey-member-scaffold",
  "criterion": "<=80.04", "source": "clip_motif_layout_no_occupancy/summary.json"},
  {"metric": "mean_physical_density", "value": 0.3000901937484741, "split": "student-storey-member-scaffold",
  "criterion": "0.30+/-0.02", "source": "clip_motif_layout_no_occupancy/summary.json"},
  {"metric": "mass_on_scaffold", "value": 0.9994097602551909, "split": "student-storey-member-scaffold",
  "source": "clip_motif_layout_no_occupancy/summary.json"}, {"metric": "clip_loss_raw",
  "value": 0.770060122013092, "split": "student-storey-member-scaffold", "source":
  "clip_motif_layout_no_occupancy/summary.json"}, {"metric": "volume_actual", "value":
  0.2930908203125, "split": "student-storey-member-scaffold", "source": "clip_motif_layout_no_occupancy/summary.json"},
  {"metric": "scaffold_mean", "value": 0.7668279409408569, "split": "student-storey-member-scaffold",
  "source": "clip_motif_layout_no_occupancy/summary.json"}, {"metric": "clip_motif_f1p0000",
  "value": 0.32919248938560486, "split": "student-storey-member-scaffold", "source":
  "clip_motif_layout_no_occupancy/summary.json"}, {"metric": "clip_motif_f0p2500",
  "value": 0.36691123247146606, "split": "student-storey-member-scaffold", "source":
  "clip_motif_layout_no_occupancy/summary.json"}, {"metric": "clip_motif_f0p0625",
  "value": 0.39332401752471924, "split": "student-storey-member-scaffold", "source":
  "clip_motif_layout_no_occupancy/summary.json"}, {"metric": "compliance", "value":
  80.04190063476562, "split": "teacher-motif-scale-no-occupancy", "source": "clip_motif_layout_no_occupancy/teacher/summary.json"},
  {"metric": "clip_loss_raw", "value": 0.747924268245697, "split": "teacher-motif-scale-no-occupancy",
  "source": "clip_motif_layout_no_occupancy/teacher/summary.json"}]'
related_to: expe-2293aefa,expe-55c7caac
---
Two-pass look at script/resources/results/clip_motif_layout_no_occupancy/. Teacher reused from this session's seed-12 motif-scale run (teacher/summary.json: compliance 80.0419, clip_loss_raw 0.7479, volume_actual 0.3220, mean density 0.3001, crop distances building/storey/member 0.3097/0.3577/0.3943) matching expe-2293aefa. Student: init from teacher, raw multi-scale CLIP on, storey+member soft scaffold annealed 4000 to 400. Converged in 57 steps (upsamples 12, 21). Recognizable jaw/teeth/skull language remains in student raw. Numeric gates held: mean physical density 0.3001, compliance 76.965 not worse than teacher 80.04. mass_on_scaffold 0.9994 at occupancy threshold 0.5; scaffold_mean 0.7668. Student clip_loss_raw 0.7701 vs teacher 0.7479; volume_actual 0.2931. Prior A/B dirs clip_motif_scale_no_occupancy and clip_motif_scale_physical_density_no_occupancy were not overwritten. Full suite 250 passed, 8 skipped.
