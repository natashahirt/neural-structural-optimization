---
id: expe-55c7caac
type: experiment
project: semantopology_hardfork
parent_id: plan-cbdb1948
title: Building-scale layout envelope fills the facade
node_label: Building-scale layout envelope fills the facade
tags: clip-scale,motif-layout,scaffold,building-envelope
status: active
open_threads: 0
success: 'false'
files: neural_structural_optimization/experiment.py@c03246a, script/resources/results/clip_motif_layout_no_occupancy/summary_building_envelope.json@c03246a
session_id: sess-4872d00d
created_at: '2026-09-05T22:34:15.637169+00:00'
updated_at: '2026-09-05T22:34:15.637169+00:00'
has_child_rationale: Building-scale layout envelopes fill the facade and cannot constrain
  member placement.
results: '[{"metric": "scaffold_mean", "value": 0.9245915412902832, "split": "student-building-storey-member-envelope",
  "source": "summary_building_envelope.json"}, {"metric": "mass_on_scaffold", "value":
  1.0, "split": "student-building-storey-member-envelope", "source": "summary_building_envelope.json"},
  {"metric": "compliance", "value": 76.94683837890625, "split": "student-building-storey-member-envelope",
  "criterion": "<=80.04", "source": "summary_building_envelope.json"}]'
---
First student used max of building+storey+member distance envelopes. On the skeletons teacher, building-scale sigma=64 px filled the facade: scaffold_mean 0.925, mass_on_scaffold 1.0, so the spatial prior was nearly vacuous. Compliance 76.95 and mean density 0.300 still held vs teacher 80.04, but that does not test layout. Default resolved_layout_scale_fracs now drops the 1.0 building fraction (CLIP still crops it); layout uses storey 0.25 and member 0.0625. Diagnostic summary kept as summary_building_envelope.json.
