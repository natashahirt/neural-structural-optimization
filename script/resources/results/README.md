# Experiment outputs

Each family is a directory (or a prefix of sibling directories). Open that
family's `comparison.png` / `summary.json`; do not mix families.

| family | what it is |
|---|---|
| `clip_saliency_compliance/` | Live CLIP/SDS occupancy prior vs exact FEA (current) |
| `SUCCESS_sketch_to_structure/` | Hand-sketch occupancy that actually redirects load paths |
| `clip_motif_scale_*` | Physical-scale CLIP crops, with/without occupancy |
| `clip_motif_layout_*` | Teacher/student layout transfer |
| `clip_dream_layout_*` | Pixel CLIP-dreams (texture / disconnected blobs) |
| `generated_stencil_*` | Rejected path: AI-generated images used as stencils |
| `stage6_*` | Sketch-to-structure ablations on the Venice golden |
| `stage2_profile_step*.json` | Solver timing |
| `250214_skeleton_loss_test_*` | Early CLIP-on-golden loss check |
