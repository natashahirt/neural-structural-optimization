---
id: note-b2a8ce44
type: note
project: semantopology_hardfork
parent_id: plan-835d10d9
title: 'Drop the perimeter/TV term from Stage A: it rewards the exact blobs that failed
  connectivity'
node_label: 'Drop the perimeter/TV term from Stage A: it reward'
tags: stage-a,connectivity,tv,perimeter,correction,e1b
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-34f63c5d
created_at: '2026-09-08T15:26:34.081401+00:00'
updated_at: '2026-09-08T15:26:34.081401+00:00'
---
Correction to the E1b plan (parent), made after actually viewing the expe-41216fc2 scaffold PNG rather than only its metrics.

The plan listed perimeter/total-variation as Stage A's secondary dial for 'consolidate into few large features'. That is wrong for this failure mode and should be dropped.

EVIDENCE: script/resources/results/clip_dream_layout_whole_butterfly_wing_venation_control16x8_smoke/ - dream_upsampled.png shows the coarse control fully solved grain (large smooth bands, contrast ratio 0.0907 vs 1.05 at full res), but scaffold.png is ~14 compact ROUNDED isolated blobs, leopard-spot pattern, no top-to-bottom span.

WHY TV IS HARMFUL HERE: TV at fixed volume minimizes boundary length, and the minimum-perimeter shape is a circle. So TV rewards exactly the compact rounded blobs that already failed the connectivity gate, and penalizes the long slender members a load path is made of. Connectivity and low perimeter are in direct tension.

WHY IT IS ALSO REDUNDANT: TV was proposed to fight hatching/texture. The coarse control grid already solves that more cheaply and more robustly (unrepresentable beats penalized). In the coarse-control regime TV is redundant at best.

REVISED STAGE A TERMS: CLIP + hard volume-fraction constraint + a spanning/connectivity term between support boundaries (rewarding elongation and contact, not compactness). Feature size is handled by the grid resolution alone. Sweep resolution as the primary dial; the CLIP-to-connectivity ratio becomes the secondary dial in place of CLIP-to-TV.

General lesson: the metrics in expe-41216fc2 said 'failed connectivity, 14 components' but the SHAPE of the failure - compact and round rather than fragmented and stringy - was only visible in the PNG, and it is the shape that determines which regularizer is correct.
