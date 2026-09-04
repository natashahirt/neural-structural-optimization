---
id: note-230aab57
type: note
project: semantopology_hardfork
parent_id: plan-265959fb
title: Volume constraint verified across volfrac/eta/beta/masks; the gate test was
  checking it wrong
node_label: Volume constraint verified across volfrac/eta/beta
tags: stage-1,volfrac,test-quality,audit
status: active
open_threads: 0
success: 'null'
files: neural_structural_optimization/tests/test_discretization.py@c97b554
session_id: sess-bfd66d57
created_at: '2026-09-04T00:35:22.439227+00:00'
updated_at: '2026-09-04T00:35:22.439227+00:00'
results: '[{"metric": "volfrac_error_projection_on_worst", "value": 1.5e-13, "unit":
  "absolute", "split": "4 problems x volfrac {0.1,0.3,0.5,0.7} x eta {0.3,0.5,0.7}
  x beta {4,16}, design-region mean", "criterion": "<=1e-9", "source": "stage-1-volfrac-audit"},
  {"metric": "volfrac_error_projection_off_worst", "value": 0.000101, "unit": "absolute",
  "split": "multistory_building volfrac=0.3, legacy pre-filter enforcement", "criterion":
  "pre-existing; pinned as a bound, not an equality", "source": "stage-1-volfrac-audit"},
  {"metric": "naive_full_array_mean_error", "value": 0.252, "unit": "absolute", "split":
  "l_shape volfrac=0.7 beta=16, density.mean() vs design-region mean", "criterion":
  "demonstrates the old assertion was unsound", "source": "stage-1-volfrac-audit"},
  {"metric": "tests_passing", "value": 68, "unit": "count", "split": "neural_structural_optimization/tests",
  "criterion": "suite green", "source": "stage-1-volfrac-audit"}]'
---
Follow-up audit after stage 1, prompted by a request to confirm volfrac actually tests correctly. THE CONSTRAINT IS SOUND; THE TEST WAS NOT.

SWEPT 4 problems x volfrac {0.1,0.3,0.5,0.7} x eta {0.3,0.5,0.7} x beta {1,4,16}. With projection ON the design-region mean matches volfrac to a worst case of 1.5e-13 across every combination, including non-trivial masks, with zero density bleed outside the mask.

TWO DEFECTS IN THE GATE TEST, both now fixed:

1. IT AVERAGED THE WRONG THING. VolumeUnderProjectionTest asserted density.mean() over the FULL array, which on a masked problem averages the zeroed non-design elements in. On l_shape at volfrac 0.7 the plain mean is off by 2.52e-01 while the true design-region mean is correct to 1e-14. The test passed only because the default problem's mask is all ones - it would have sailed through a real violation on any masked problem. Added a _design_region_mean helper and a test that pins WHY it exists so nobody simplifies it back.

2. IT SAT ON THE BLIND SPOT. The only volume assertion ran at volfrac == eta == 0.5, which is exactly the mean-neutral point where the projection cannot shift the mean - the same operating point that let the earlier P0 escape two reviews. Added sweeps off the diagonal and over masked problems.

Suite went 63 -> 68 tests, 60 -> 92 subtests. Default-path parity re-verified byte-identical.

SEPARATE FINDING, projection OFF: the legacy path misses volfrac by up to 1.01e-04 (multistory_building volfrac=0.3), not the 1.6e-5 measured earlier on the default problem. Pre-existing, caused by enforcing volume on the raw sigmoid before a filter that is not mean-preserving. Pinned as an upper BOUND rather than an equality, deliberately, because stage 3's Venice reproduction depends on this path not moving.

BENIGN ISSUE NOTED, NOT FIXED: normalized_cone_filter_matrix does weights = 1/raw_filters.sum(axis=0) and emits a divide-by-zero RuntimeWarning on l_shape. Output stays finite - the infinite weights land on masked-out entries and are multiplied by zero - and the bleed test confirms density is exactly 0 outside the mask. Pre-existing and untouched by stage 1, but it is an inf that happens to cancel rather than one that cannot arise.
