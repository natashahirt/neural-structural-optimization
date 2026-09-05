"""Run Venice 250214 with sketch 12 as a spatial mass prior.

Keeps the golden problem, CLIP, seed, and AdaptiveAdam schedule. Seeds ``z``
from occupancy union load pixels (not thick_outer_lins), supports a sweep of
the final spatial weight, and can add stage-scheduled motif recurrence.

    PYTHONPATH="$PWD" python script/stage6_sketch_on_golden.py

Writes labeled panels under ``script/resources/results/stage6_sketch12_init_anneal/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization.experiment import GOLDEN, SketchConfig
from neural_structural_optimization.models.loss_sketch import (
    SKETCH_DIR,
    apply_sketch_config,
    load_site_mask,
    load_sketch_occupancy,
    mass_fraction_on_occupancy,
    sketch_mass_prior_loss,
    sketch_motif_loss,
    sketch_patch_vocabulary_loss,
)

SKETCH_REL = SKETCH_DIR / '12.jpg'
# Venice CLIP weight is compliance * 10: ~4900 at step 0, ~740 at
# convergence. A constant 400 never competed. Start at CLIP's early scale.
DEFAULT_SKETCH_WEIGHT = 4000.0
DEFAULT_SKETCH_WEIGHT_END = 400.0
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / 'script' / 'resources' / 'results' / 'stage6_sketch12_init_anneal')
LOADPIXELS_RUN_PATH = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / 'stage6_sketch12_loadpixels' / 'sketch_run.png')
GOLDEN_FINAL_IMAGE_PATH = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / '250214_skeleton_loss_test_balanced_dynamic' / '2_final'
    / ('0_final-P_multistory_building-M_Ada-T_skeletons-W_128-H_256-V_0.30'
       '-LR_0.20-CW_10-ID_01-CompL_74.00-ClipL_0.38-VA_0.31'
       '-balanced_dynamic.jpg'))


def _load_golden_script():
    script = REPO_ROOT / 'script' / 'venice_golden_250214.py'
    spec = importlib.util.spec_from_file_location('venice_golden_250214', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ink_black(field: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Occupancy/density panel: 1 ? black (material), sized to the reference JPEG."""
    arr = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(
        (255.0 * (1.0 - arr)).clip(0, 255).astype(np.uint8), mode='L')
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def _venice_display(raw: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Same round trip as the golden JPEG: short-edge resize, clamp, invert."""
    import torch
    from neural_structural_optimization.models.loss_clip import _resize_short_side

    resized = _resize_short_side(
        torch.as_tensor(np.ascontiguousarray(raw, dtype=np.float32))[None, None],
        GOLDEN.clip_resize_short_side,
    ).clamp(0.0, 1.0)
    image = Image.fromarray(
        (255.0 * (1.0 - resized[0, 0].cpu().numpy())).clip(0, 255).astype(np.uint8),
        mode='L',
    )
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def _occupancy() -> np.ndarray:
    return load_sketch_occupancy(
        REPO_ROOT / SKETCH_REL,
        height=GOLDEN.height,
        width=GOLDEN.width,
    )


def _hstack(panels: list[tuple[str, Image.Image]], path: Path) -> Path:
    gap, label_h = 16, 28
    images = [im.convert('L') for _, im in panels]
    width, height = images[0].size
    aligned = [
        im if im.size == (width, height)
        else im.resize((width, height), Image.Resampling.NEAREST)
        for im in images
    ]
    canvas = Image.new(
        'L',
        (width * len(aligned) + gap * (len(aligned) - 1), height + label_h),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (title, _), im in zip(panels, aligned):
        canvas.paste(im, (x, label_h))
        draw.text((x + 8, 6), title, fill=0)
        x += width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def save_occupancy_preview(output_dir: Path) -> dict[str, Path]:
    occupancy = _occupancy()
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    occ_img = _ink_black(occupancy, reference.size)
    output_dir.mkdir(parents=True, exist_ok=True)
    occ_path = output_dir / 'occupancy.png'
    occ_img.save(occ_path)
    preview = _hstack(
        [
            ('Sketch 12 occupancy (ink black)', occ_img),
            ('Venice 250214 reference', reference),
        ],
        output_dir / 'occupancy_vs_reference.png',
    )
    return {'occupancy': occ_path, 'preview': preview}


def run_sketch_on_golden(
    *,
    sketch_weight_end: float = DEFAULT_SKETCH_WEIGHT_END,
    motif_weight: float = 0.0,
    motif_weight_end: float | None = None,
    motif_scales: tuple[int, ...] = (1, 2, 4),
    patch_weight: float = 0.0,
    patch_weight_end: float | None = None,
    patch_sizes: tuple[int, ...] = (7, 15),
    patch_stride: int = 2,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    comparisons: tuple[tuple[str, Path], ...] = (),
) -> dict:
    import torch

    from neural_structural_optimization import configure_torch_threads

    golden = _load_golden_script()
    configure_torch_threads()
    clip_loss = golden.build_clip_loss(GOLDEN)
    golden.seed_everything(GOLDEN.seed)
    model = golden.build_model(clip_loss, GOLDEN)
    apply_sketch_config(
        model,
        SketchConfig(
            path=str(SKETCH_REL),
            weight=DEFAULT_SKETCH_WEIGHT,
            weight_end=sketch_weight_end,
            init_from_occupancy=True,
            motif_weight=motif_weight,
            motif_weight_end=motif_weight_end,
            motif_scales=motif_scales,
            patch_weight=patch_weight,
            patch_weight_end=patch_weight_end,
            patch_sizes=patch_sizes,
            patch_stride=patch_stride,
        ),
        height=GOLDEN.height,
        width=GOLDEN.width,
        repo_root=REPO_ROOT,
    )
    golden.seed_everything(GOLDEN.seed)
    ds = golden.attach_final_raw_design(
        golden.build_optimizer(model, GOLDEN).optimize(), model)

    occupancy = _occupancy()
    density = model.get_physical_density(model()).detach().cpu().numpy()
    if density.ndim == 3:
        density = density[0]
    load_sites = load_site_mask(
        model.env.args['forces'],
        nely=int(model.env.args['nely']),
        nelx=int(model.env.args['nelx']),
    )
    allowed = np.maximum(occupancy, load_sites)
    raw = np.ascontiguousarray(ds['final_design_raw'].values, dtype=np.float32)
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    size = reference.size
    occ_img = _ink_black(occupancy, size)
    allowed_img = _ink_black(allowed, size)
    density_img = _ink_black(np.clip(density, 0.0, 1.0), size)
    replay = _venice_display(raw, size)
    loadpixels_img = (
        Image.open(LOADPIXELS_RUN_PATH).convert('L').resize(size, Image.Resampling.NEAREST)
        if LOADPIXELS_RUN_PATH.is_file() else None)

    replay_path = output_dir / 'sketch_run.png'
    density_path = output_dir / 'physical_density.png'
    allowed_path = output_dir / 'allowed_template.png'
    replay.save(replay_path)
    density_img.save(density_path)
    allowed_img.save(allowed_path)
    comparison_panels = [
        ('Sketch 12 occupancy', occ_img),
        ('Allowed (occ union loads)', allowed_img),
    ]
    for title, path in comparisons:
        if path.is_file():
            comparison_panels.append((
                title,
                Image.open(path).convert('L').resize(
                    size, Image.Resampling.NEAREST),
            ))
    if patch_weight:
        run_title = f'Global {sketch_weight_end:g} + patch motif'
    elif motif_weight:
        run_title = f'Global {sketch_weight_end:g} + motif'
    else:
        run_title = f'Global end {sketch_weight_end:g}'
    comparison_panels.append((run_title, replay))
    if loadpixels_img is not None and not comparisons:
        comparison_panels.append(('Load-pixel-only baseline', loadpixels_img))
    comparison_panels.append(('Venice 250214 reference', reference))
    comparison = _hstack(comparison_panels, output_dir / 'comparison.png')
    density_strip = _hstack(
        [
            ('Sketch 12 occupancy', occ_img),
            ('Allowed (occ union loads)', allowed_img),
            ('Physical density (volfrac held)', density_img),
            ('Venice 250214 reference', reference),
        ],
        output_dir / 'density_vs_reference.png',
    )
    density_t = torch.as_tensor(density, dtype=torch.float32)
    occupancy_t = torch.as_tensor(occupancy, dtype=torch.float32)
    load_sites_t = torch.as_tensor(load_sites, dtype=torch.float32)
    spatial_mass_loss = float(sketch_mass_prior_loss(
        density_t, occupancy_t, load_sites=load_sites_t))
    motif_loss = float(sketch_motif_loss(
        density_t, occupancy_t, scales=motif_scales))
    patch_loss = float(sketch_patch_vocabulary_loss(
        density_t,
        occupancy_t,
        load_sites=load_sites_t,
        patch_sizes=patch_sizes,
        stride=patch_stride,
    ))
    summary = {
        'steps': int(ds.sizes['step']),
        'converged': bool(ds.attrs['converged']),
        'resize_steps': [int(s) for s in ds.attrs['resize_steps']],
        'sketch_weight_start': DEFAULT_SKETCH_WEIGHT,
        'sketch_weight_end': sketch_weight_end,
        'init_from_occupancy': True,
        'motif_weight_peak': motif_weight,
        'motif_weight_end': motif_weight_end,
        'motif_scales': list(motif_scales),
        'patch_weight_peak': patch_weight,
        'patch_weight_end': patch_weight_end,
        'patch_sizes': list(patch_sizes),
        'patch_stride': patch_stride,
        'mass_on_occupancy': mass_fraction_on_occupancy(density, occupancy),
        'mass_on_allowed': mass_fraction_on_occupancy(density, allowed),
        'spatial_mass_loss': spatial_mass_loss,
        'weighted_spatial_mass_loss': sketch_weight_end * spatial_mass_loss,
        'motif_loss': motif_loss,
        'weighted_motif_loss': (
            (motif_weight if motif_weight_end is None else motif_weight_end)
            * motif_loss),
        'patch_loss': patch_loss,
        'weighted_patch_loss': (
            (patch_weight if patch_weight_end is None else patch_weight_end)
            * patch_loss),
        'load_site_frac': float(load_sites.mean()),
        'allowed_frac': float(allowed.mean()),
        'mean_physical_density': float(np.mean(density)),
        'volume_actual': golden.venice_volume_ratio(raw),
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': float(ds['clip_loss'][-1]),
        'total_loss': float(ds['loss'][-1]),
        'paths': {
            'replay': str(replay_path),
            'density': str(density_path),
            'allowed': str(allowed_path),
            'comparison': str(comparison),
            'density_strip': str(density_strip),
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--occupancy-only',
        action='store_true',
        help='write occupancy vs reference and exit (no CLIP, no FEA)',
    )
    parser.add_argument(
        '--weight-end', type=float, default=DEFAULT_SKETCH_WEIGHT_END)
    parser.add_argument('--motif-weight', type=float, default=0.0)
    parser.add_argument('--motif-weight-end', type=float)
    parser.add_argument(
        '--motif-scales',
        default='1,2,4',
        help='comma-separated pooling scales for the motif descriptor',
    )
    parser.add_argument('--patch-weight', type=float, default=0.0)
    parser.add_argument('--patch-weight-end', type=float)
    parser.add_argument(
        '--patch-sizes',
        default='7,15',
        help='comma-separated odd patch widths',
    )
    parser.add_argument('--patch-stride', type=int, default=2)
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='result directory (relative paths resolve from the repository root)',
    )
    parser.add_argument(
        '--compare',
        action='append',
        default=[],
        metavar='LABEL=PATH',
        help='add a labeled prior run to comparison.png',
    )
    args = parser.parse_args(argv)
    if (args.weight_end < 0 or args.motif_weight < 0
            or args.patch_weight < 0):
        parser.error('weights must be non-negative')
    motif_scales = tuple(
        int(value.strip()) for value in args.motif_scales.split(',')
        if value.strip())
    if not motif_scales or any(scale < 1 for scale in motif_scales):
        parser.error('--motif-scales must contain positive integers')
    patch_sizes = tuple(
        int(value.strip()) for value in args.patch_sizes.split(',')
        if value.strip())
    if (not patch_sizes
            or any(size < 3 or size % 2 == 0 for size in patch_sizes)):
        parser.error('--patch-sizes must contain odd integers >= 3')
    if args.patch_stride < 1:
        parser.error('--patch-stride must be positive')
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    comparisons = []
    for item in args.compare:
        if '=' not in item:
            parser.error('--compare must be LABEL=PATH')
        label, raw_path = item.split('=', 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        comparisons.append((label, path))
    paths = save_occupancy_preview(output_dir)
    print(f'Wrote occupancy to {paths["occupancy"]}')
    print(f'Wrote occupancy vs reference to {paths["preview"]}')
    if args.occupancy_only:
        return 0
    print(
        'Running Venice 250214 + sketch 12 '
        f'(init occupancy, {DEFAULT_SKETCH_WEIGHT}->{args.weight_end}, '
        f'motif peak={args.motif_weight}, patch peak={args.patch_weight})...')
    summary = run_sketch_on_golden(
        sketch_weight_end=args.weight_end,
        motif_weight=args.motif_weight,
        motif_weight_end=args.motif_weight_end,
        motif_scales=motif_scales,
        patch_weight=args.patch_weight,
        patch_weight_end=args.patch_weight_end,
        patch_sizes=patch_sizes,
        patch_stride=args.patch_stride,
        output_dir=output_dir,
        comparisons=tuple(comparisons),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
