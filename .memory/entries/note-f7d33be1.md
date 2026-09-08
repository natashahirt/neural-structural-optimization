---
id: note-f7d33be1
type: note
project: semantopology_hardfork
parent_id: deci-87423b3f
title: 'Parked: bold filled CLIP shape, then edge-detect into a sketch prior'
node_label: 'Parked: bold filled CLIP shape, then edge-detect i'
tags: parked,edge-detection,silhouette,sketch-prior,stage-a
status: active
open_threads: 0
success: 'null'
files: ''
session_id: sess-0bd655dd
created_at: '2026-09-08T17:24:46.113481+00:00'
updated_at: '2026-09-08T17:24:46.113481+00:00'
---
PARKED until Stage A produces a good filled skull. Do not implement yet.

Thought: get CLIP to make a BOLD FILLED shape, then convert it to a sketch prior by EDGE DETECTION rather than asking CLIP for line-art/outline.

WHY: CLIP is good at filled high-contrast shapes (the 32x16 and 48x24 dreams) and bad at thin strokes (hatching is the live failure mode). The Stage 6 sketches that actually worked (SUCCESS_sketch_to_structure allowed_template.png) are LINE DRAWINGS with thick strokes, not filled blobs. So the conversion CLIP-can-do -> format-physics-already-eats is: filled binary silhouette -> edge detect -> (optional dilation to member width) -> occupancy prior.

WHAT EDGE DETECTION DOES AND DOES NOT DO:
- Traces existing holes (orbits, nasal cavity) as inner contours IF they exist in the filled field. It cannot invent facial features that the silhouette lacks.
- Raw Canny/Sobel is 1px; must dilate to member width or physics has nothing to grab. loss_sketch.py already has distance_transform_edt for envelopes.
- Filled occupancy = 'put mass in the skull interior'. Edged occupancy = 'put mass on the skull FRAME'. The frame is the more structural object and closer to the human sketches.

PROMPT IMPLICATION: wrapper vocabulary should bias toward filled styles (silhouette / clip-art / pictogram), NOT outline/line-art/woodcut, because those name the thin-stroke failure mode. Edge detection is the stroke conversion, applied after CLIP, so the prompt should not also ask for strokes.

This is a geometric post-process on CLIP's own output, not an AI-generated stencil image (anno-4676f76b still holds).

CURRENT PRIORITY (user): get the filled shape good first via in-loop heaviside + volume + scaffold passthrough. Revisit this conversion once a silhouette reads as a skull.
