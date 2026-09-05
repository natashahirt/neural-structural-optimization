"""Run Venice 250214 with a sketch as a spatial mass prior.

Keeps the golden problem, CLIP, seed, and AdaptiveAdam schedule. Seeds ``z``
from occupancy union load pixels and anneals the spatial prior from 4000 at
the coarse stage to 400 at full resolution. Motif and patch terms stay off
unless explicitly requested.

    PYTHONPATH="$PWD" python script/stage6_sketch_on_golden.py --sketch 12
    PYTHONPATH="$PWD" python script/stage6_sketch_on_golden.py --corpus

Writes labeled panels under ``script/resources/results/stage6_sketch<id>_init_anneal/``.
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
    SKETCH_CORPUS,
    SKETCH_DIR,
    apply_sketch_config,
    load_site_mask,
    load_sketch_occupancy,
    mass_fraction_on_occupancy,
    sketch_mass_prior_loss,
)

# Venice CLIP weight is compliance * 10: ~4900 at step 0, ~740 at
# convergence. A constant 400 never competed. Start at CLIP's early scale.
DEFAULT_SKETCH_WEIGHT = 4000.0
DEFAULT_SKETCH_WEIGHT_END = 400.0
DEFAULT_SKETCH = '12.jpg'
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


def _sketch_name(raw: str) -> str:
    """Accept ``12``, ``12.jpg``, or a path under the corpus directory."""
    name = Path(raw).name
    if '.' not in name:
        name = f'{name}.jpg'
    if name not in SKETCH_CORPUS:
        raise ValueError(
            f'unknown sketch {raw!r}; expected one of {SKETCH_CORPUS}')
    return name


def _sketch_stem(name: str) -> str:
    return Path(name).stem


def _sketch_rel(name: str) -> Path:
    return SKETCH_DIR / name


def _output_dir_for(name: str) -> Path:
    return (
        REPO_ROOT / 'script' / 'resources' / 'results'
        / f'stage6_sketch{_sketch_stem(name)}_init_anneal')


def _occupancy(name: str) -> np.ndarray:
    return load_sketch_occupancy(
        REPO_ROOT / _sketch_rel(name),
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


def save_occupancy_preview(output_dir: Path, sketch_name: str) -> dict[str, Path]:
    occupancy = _occupancy(sketch_name)
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    occ_img = _ink_black(occupancy, reference.size)
    output_dir.mkdir(parents=True, exist_ok=True)
    occ_path = output_dir / 'occupancy.png'
    occ_img.save(occ_path)
    stem = _sketch_stem(sketch_name)
    preview = _hstack(
        [
            (f'Sketch {stem} occupancy (ink black)', occ_img),
            ('Venice 250214 skeletons reference', reference),
        ],
        output_dir / 'occupancy_vs_reference.png',
    )
    return {'occupancy': occ_path, 'preview': preview}


def run_sketch_on_golden(
    *,
    sketch_name: str = DEFAULT_SKETCH,
    sketch_weight_end: float = DEFAULT_SKETCH_WEIGHT_END,
    output_dir: Path | None = None,
    comparisons: tuple[tuple[str, Path], ...] = (),
) -> dict:
    import torch

    from neural_structural_optimization import configure_torch_threads

    sketch_name = _sketch_name(sketch_name)
    stem = _sketch_stem(sketch_name)
    if output_dir is None:
        output_dir = _output_dir_for(sketch_name)

    golden = _load_golden_script()
    configure_torch_threads()
    clip_loss = golden.build_clip_loss(GOLDEN)
    golden.seed_everything(GOLDEN.seed)
    model = golden.build_model(clip_loss, GOLDEN)
    apply_sketch_config(
        model,
        SketchConfig(
            path=str(_sketch_rel(sketch_name)),
            weight=DEFAULT_SKETCH_WEIGHT,
            weight_end=sketch_weight_end,
            init_from_occupancy=True,
        ),
        height=GOLDEN.height,
        width=GOLDEN.width,
        repo_root=REPO_ROOT,
    )
    golden.seed_everything(GOLDEN.seed)
    ds = golden.attach_final_raw_design(
        golden.build_optimizer(model, GOLDEN).optimize(), model)

    occupancy = _occupancy(sketch_name)
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
        (f'Sketch {stem} occupancy', occ_img),
        ('Allowed (occ union loads)', allowed_img),
    ]
    for title, path in comparisons:
        if path.is_file():
            comparison_panels.append((
                title,
                Image.open(path).convert('L').resize(
                    size, Image.Resampling.NEAREST),
            ))
    comparison_panels.append((f'Sketch {stem} + skeletons', replay))
    if loadpixels_img is not None and not comparisons:
        comparison_panels.append(('Load-pixel-only baseline', loadpixels_img))
    comparison_panels.append(('Venice 250214 skeletons', reference))
    comparison = _hstack(comparison_panels, output_dir / 'comparison.png')
    density_strip = _hstack(
        [
            (f'Sketch {stem} occupancy', occ_img),
            ('Allowed (occ union loads)', allowed_img),
            ('Physical density (volfrac held)', density_img),
            ('Venice 250214 skeletons', reference),
        ],
        output_dir / 'density_vs_reference.png',
    )
    density_t = torch.as_tensor(density, dtype=torch.float32)
    occupancy_t = torch.as_tensor(occupancy, dtype=torch.float32)
    load_sites_t = torch.as_tensor(load_sites, dtype=torch.float32)
    spatial_mass_loss = float(sketch_mass_prior_loss(
        density_t, occupancy_t, load_sites=load_sites_t))
    summary = {
        'sketch': sketch_name,
        'clip_prompt': GOLDEN.prompt,
        'steps': int(ds.sizes['step']),
        'converged': bool(ds.attrs['converged']),
        'resize_steps': [int(s) for s in ds.attrs['resize_steps']],
        'sketch_weight_start': DEFAULT_SKETCH_WEIGHT,
        'sketch_weight_end': sketch_weight_end,
        'init_from_occupancy': True,
        'mass_on_occupancy': mass_fraction_on_occupancy(density, occupancy),
        'mass_on_allowed': mass_fraction_on_occupancy(density, allowed),
        'spatial_mass_loss': spatial_mass_loss,
        'weighted_spatial_mass_loss': sketch_weight_end * spatial_mass_loss,
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


def save_corpus_strip(sketches: tuple[str, ...], output_dir: Path) -> Path:
    """Occupancy | result for each finished sketch, plus the skeletons reference."""
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    size = reference.size
    panels: list[tuple[str, Image.Image]] = []
    for name in sketches:
        stem = _sketch_stem(name)
        occ = _ink_black(_occupancy(name), size)
        run_path = _output_dir_for(name) / 'sketch_run.png'
        if not run_path.is_file():
            continue
        result = Image.open(run_path).convert('L').resize(
            size, Image.Resampling.NEAREST)
        panels.append((f'{stem} occupancy', occ))
        panels.append((f'{stem} + skeletons', result))
    panels.append(('Venice 250214 skeletons', reference))
    return _hstack(panels, output_dir / 'corpus_comparison.png')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--occupancy-only',
        action='store_true',
        help='write occupancy vs reference and exit (no CLIP, no FEA)',
    )
    parser.add_argument(
        '--sketch',
        default=DEFAULT_SKETCH,
        help='corpus sketch id or filename (default: 12.jpg)',
    )
    parser.add_argument(
        '--corpus',
        action='store_true',
        help='run every corpus sketch except 12 (already the selected look)',
    )
    parser.add_argument(
        '--weight-end', type=float, default=DEFAULT_SKETCH_WEIGHT_END)
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
    if args.weight_end < 0:
        parser.error('weights must be non-negative')
    try:
        sketches = (
            tuple(name for name in SKETCH_CORPUS if name != DEFAULT_SKETCH)
            if args.corpus else (_sketch_name(args.sketch),)
        )
    except ValueError as exc:
        parser.error(str(exc))
    comparisons = []
    for item in args.compare:
        if '=' not in item:
            parser.error('--compare must be LABEL=PATH')
        label, raw_path = item.split('=', 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        comparisons.append((label, path))
    summaries = []
    for name in sketches:
        output_dir = args.output_dir or _output_dir_for(name)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        paths = save_occupancy_preview(output_dir, name)
        print(f'Wrote occupancy to {paths["occupancy"]}')
        print(f'Wrote occupancy vs reference to {paths["preview"]}')
        if args.occupancy_only:
            continue
        print(
            f'Running Venice 250214 skeletons + sketch {Path(name).stem} '
            f'(init occupancy, {DEFAULT_SKETCH_WEIGHT}->{args.weight_end})...')
        summary = run_sketch_on_golden(
            sketch_name=name,
            sketch_weight_end=args.weight_end,
            output_dir=output_dir,
            comparisons=tuple(comparisons),
        )
        print(json.dumps(summary, indent=2))
        summaries.append(summary)
    if args.corpus and not args.occupancy_only:
        corpus_dir = (
            REPO_ROOT / 'script' / 'resources' / 'results'
            / 'stage6_corpus_init_anneal')
        corpus_dir.mkdir(parents=True, exist_ok=True)
        strip = save_corpus_strip(
            (DEFAULT_SKETCH,) + sketches, corpus_dir)
        catalog = {
            'clip_prompt': GOLDEN.prompt,
            'sketch_weight_start': DEFAULT_SKETCH_WEIGHT,
            'sketch_weight_end': args.weight_end,
            'init_from_occupancy': True,
            'runs': summaries,
            'corpus_strip': str(strip),
        }
        (corpus_dir / 'summary.json').write_text(
            json.dumps(catalog, indent=2) + '\n')
        print(f'Wrote corpus comparison to {strip}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
