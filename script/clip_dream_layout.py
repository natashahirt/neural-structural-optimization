"""E1: CLIP dreams a layout; physics builds on that occupancy.

The pixel motif-scale runs and the layout-distillation student all started from
a field that had already seen compliance. Their scaffolds therefore endorsed
the physics load path (spatial_mass_loss 0.0078 vs Stage 6's 0.31-0.60). This
script inverts the order:

1. Dream: AdaptivePixel at the coarse 32x64 grid, CLIP on the *whole
   elevation* only (Venice RandomResizedCrop; no storey/member crops),
   no FEA. Load-site collectors stay off during the dream so floors are
   not frames; they are unioned when the mass prior is applied.
2. Upsample that one drawing to 128x256 and threshold it like a sketch
   (rank ink, no storey envelope) so occupancy is one connected elevation.
3. Gate: score a physics-layout proxy against that occupancy. If
   ``spatial_mass_loss < 0.25`` the prior is tautological again - stop.
4. Physics: Venice AdaptiveAdam + the same whole-building CLIP, Stage 6
   occupancy recipe (init from occupancy union load pixels, anneal
   4000 to 400). Not seeded from a physics teacher.

    PYTHONPATH="$PWD" python script/clip_dream_layout.py --dream-only
    PYTHONPATH="$PWD" python script/clip_dream_layout.py

E2 (CNN basis) and E3 (pretrained decoder) are parked in memory; do not start
them from this script.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import configure_torch_threads
from neural_structural_optimization.experiment import VeniceGoldenConfig
from neural_structural_optimization.models.loss_sketch import (
    DEFAULT_SCAFFOLD_ALLOWED_MEAN,
    DEFAULT_TAUTOLOGY_MIN_MASS_OFF,
    apply_scaffold_as_occupancy_prior,
    load_site_mask,
    mass_fraction_on_occupancy,
    motif_layout_threshold_for_allowed_mean,
    rank_ink_from_raw,
    resample_field,
    scaffold_spatial_mass_loss,
)

DEFAULT_PROMPT = 'butterfly wing venation'
DEFAULT_DREAM_STEPS = 64
DEFAULT_DREAM_LR = 0.2
DEFAULT_PHYSICS_PROXY = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / 'clip_motif_layout_no_occupancy' / 'teacher' / 'physical_density.npy')


def _load_golden():
    script = REPO_ROOT / 'script' / 'venice_golden_250214.py'
    spec = importlib.util.spec_from_file_location('venice_golden_250214', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dream_results_dir(prompt: str, repo_root: Path = REPO_ROOT, golden=None) -> Path:
    golden = golden or _load_golden()
    return (
        repo_root / 'script' / 'resources' / 'results'
        / f'clip_dream_layout_whole_{golden.prompt_slug(prompt)}')


def whole_building_dream_config(prompt: str, golden) -> VeniceGoldenConfig:
    """Motif-scale prompt, but CLIP sees the whole elevation only.

    ``motif_scale_fracs=()`` drops storey/member crops. ``union_load_sites``
    is False so floor collectors are not painted as frames; they are unioned
    into occupancy when the mass prior is applied.
    """
    config = golden.motif_scale_run_config(prompt)
    return dataclasses.replace(
        config,
        motif_scale_fracs=(),
        union_load_sites=False,
    )


def build_dream_model(config: VeniceGoldenConfig, golden):
    """Coarse AdaptivePixel + Venice CLIP path, *without* the Venice algebra.

    The algebra always solves FEA. The dream must not. ``venice_compat`` on
    CLIPLoss is enough for ``get_semantic_loss`` to see raw logits.
    """
    clip_loss = golden.build_clip_loss(config)
    golden.seed_everything(config.seed)
    model = golden.build_model(clip_loss, config)
    model.venice_loss_algebra = None
    golden.seed_everything(config.seed)
    return model


def run_clip_dream(
    model,
    *,
    steps: int,
    lr: float,
    control_height: int = 16,
    control_width: int = 8,
) -> tuple[list[float], torch.Tensor]:
    """Optimize a low-dimensional field decoded smoothly to the model grid.

    The 16x8 control has 128 variables instead of 2,048 independent pixels.
    Bilinear expansion plus one local average makes coherent regions cheap and
    removes the single-pixel texture move that dominated the direct dream.
    """
    if control_height < 2 or control_width < 2:
        raise ValueError('control grid dimensions must both be >= 2')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(model.seed))
    control = torch.full(
        (1, 1, control_height, control_width),
        float(model.env.args['volfrac']),
        device=model.device,
    )
    noise = torch.rand(control.shape, generator=generator)
    control = torch.nn.Parameter(
        control + 0.01 * (2.0 * noise.to(model.device) - 1.0))
    optimizer = torch.optim.Adam([control], lr=lr)
    losses = []
    for _ in tqdm(range(int(steps)), desc='CLIP dream (no FEA)'):
        optimizer.zero_grad(set_to_none=True)
        logits = F.interpolate(
            control,
            size=model.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        logits = F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
        logits = logits[:, 0]
        loss = model.get_semantic_loss(logits)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses, logits.detach()


def extract_dream_scaffold(
    dream_field: np.ndarray,
    *,
    height: int,
    width: int,
    target_mean: float = DEFAULT_SCAFFOLD_ALLOWED_MEAN,
):
    """Upsample the coarse dream and cut one elevation occupancy.

    Rank ink (logits are unbounded) and no distance envelope: the occupancy
    is the drawing, the way a Stage 6 sketch is one picture of the building.
    ``scale_fracs=(1.0,)`` is required by the extractor and unused at
    ``envelope_sigma_frac=0``.
    """
    upsampled = resample_field(dream_field, height, width)
    threshold, scaffold = motif_layout_threshold_for_allowed_mean(
        upsampled,
        scale_fracs=(1.0,),
        envelope_sigma_frac=0.0,
        ink_mode='rank',
        target_mean=target_mean,
    )
    return upsampled, threshold, scaffold


def evaluate_tautology_gate(
    scaffold: np.ndarray,
    density: np.ndarray,
    load_sites: Optional[np.ndarray] = None,
    *,
    min_mass_off: float = DEFAULT_TAUTOLOGY_MIN_MASS_OFF,
) -> dict:
    """Score a physics-layout proxy against the dream scaffold.

    Pass means the proxy puts at least ``min_mass_off`` of its mass *off*
    the allowed template, i.e. the scaffold is not the load path physics
    already wanted.
    """
    mass_off = scaffold_spatial_mass_loss(density, scaffold, load_sites)
    mass_on = mass_fraction_on_occupancy(density, scaffold)
    passed = float(mass_off) >= float(min_mass_off)
    return {
        'spatial_mass_loss': float(mass_off),
        'mass_on_scaffold': float(mass_on),
        'scaffold_mean': float(np.mean(scaffold)),
        'min_mass_off': float(min_mass_off),
        'passed': bool(passed),
    }


def evaluate_dream_geometry(
    field: np.ndarray,
    occupancy: np.ndarray,
    *,
    max_neighbor_contrast_ratio: float = 0.5,
) -> dict:
    """Check that a dream is smooth and its occupancy spans the elevation."""
    from scipy import ndimage

    arr = np.asarray(field, dtype=np.float64)
    std = max(float(arr.std()), 1e-12)
    dx = float(np.abs(np.diff(arr, axis=1)).mean())
    dy = float(np.abs(np.diff(arr, axis=0)).mean())
    contrast_ratio = max(dx, dy) / std

    binary = np.asarray(occupancy) >= 0.5
    labels, count = ndimage.label(binary)
    spanning_label = 0
    top = set(labels[0][labels[0] > 0].tolist())
    bottom = set(labels[-1][labels[-1] > 0].tolist())
    shared = top & bottom
    if shared:
        spanning_label = max(
            shared, key=lambda value: int(np.count_nonzero(labels == value)))
    spanning_fraction = (
        float(np.count_nonzero(labels == spanning_label)) / binary.size
        if spanning_label else 0.0)
    smooth = contrast_ratio <= float(max_neighbor_contrast_ratio)
    return {
        'neighbor_contrast_ratio': contrast_ratio,
        'max_neighbor_contrast_ratio': float(max_neighbor_contrast_ratio),
        'smooth': bool(smooth),
        'component_count': int(count),
        'top_to_bottom_connected': bool(spanning_label),
        'spanning_component_fraction': spanning_fraction,
        'passed': bool(smooth and spanning_label),
    }


def _save_field_png(path: Path, field: np.ndarray, *, rank: bool = False) -> Path:
    from PIL import Image

    arr = np.asarray(field, dtype=np.float64)
    if rank:
        arr = rank_ink_from_raw(arr).astype(np.float64)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray(
        (255.0 * (1.0 - arr)).clip(0, 255).astype(np.uint8), mode='L',
    ).save(path)
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'CLIP-only coarse dream of the whole elevation, then occupancy '
            'and Stage 6 physics. Stops before physics if the tautology gate '
            'fails.'))
    parser.add_argument(
        '--prompt', default=DEFAULT_PROMPT,
        help=f'CLIP text prompt (default {DEFAULT_PROMPT!r})')
    parser.add_argument(
        '--dream-steps', type=int, default=DEFAULT_DREAM_STEPS)
    parser.add_argument(
        '--dream-lr', type=float, default=DEFAULT_DREAM_LR)
    parser.add_argument('--control-height', type=int, default=16)
    parser.add_argument('--control-width', type=int, default=8)
    parser.add_argument(
        '--target-allowed-mean', type=float,
        default=DEFAULT_SCAFFOLD_ALLOWED_MEAN)
    parser.add_argument(
        '--min-mass-off', type=float, default=DEFAULT_TAUTOLOGY_MIN_MASS_OFF)
    parser.add_argument(
        '--baseline', type=Path, default=DEFAULT_PHYSICS_PROXY,
        help=(
            'physics-layout proxy .npy for the tautology gate. Default is '
            'the layout-distillation teacher physical density (three-bay '
            'frame). Not a structure-only control; user-confirmed as physics.'))
    parser.add_argument(
        '--dream-only', action='store_true',
        help='dream + scaffold + gate; do not run physics')
    parser.add_argument(
        '--skip-gate', action='store_true',
        help='run physics even if the tautology gate fails (debug only)')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)

    golden = _load_golden()
    config = whole_building_dream_config(args.prompt, golden)
    output_dir = args.output_dir or dream_results_dir(config.prompt, golden=golden)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.motif_scale_fracs:
        raise ValueError(
            f'whole-building dream requires empty motif_scale_fracs, got '
            f'{config.motif_scale_fracs}')

    configure_torch_threads()
    print(
        f'CLIP dream of the whole elevation, prompt={config.prompt!r}, '
        f'{args.dream_steps} steps, lr={args.dream_lr:g}, no FEA, '
        f'{args.control_height}x{args.control_width} smooth control, '
        'no motif-scale crops, no load-site frames.')
    model = build_dream_model(config, golden)
    dream_losses, dream_logits = run_clip_dream(
        model,
        steps=args.dream_steps,
        lr=args.dream_lr,
        control_height=args.control_height,
        control_width=args.control_width,
    )

    dream_coarse = np.asarray(
        dream_logits.cpu().numpy(), dtype=np.float32)
    while dream_coarse.ndim > 2:
        dream_coarse = dream_coarse[0]
    np.save(output_dir / 'dream_coarse.npy', dream_coarse)
    _save_field_png(output_dir / 'dream_coarse.png', dream_coarse, rank=True)

    upsampled, threshold, scaffold = extract_dream_scaffold(
        dream_coarse,
        height=config.height,
        width=config.width,
        target_mean=args.target_allowed_mean,
    )
    np.save(output_dir / 'dream_upsampled.npy', upsampled)
    np.save(output_dir / 'scaffold.npy', scaffold)
    _save_field_png(output_dir / 'dream_upsampled.png', upsampled, rank=True)
    _save_field_png(output_dir / 'scaffold.png', scaffold)
    geometry = evaluate_dream_geometry(upsampled, scaffold)
    print(json.dumps({'dream_geometry': geometry}, indent=2))

    baseline_path = args.baseline
    if not baseline_path.is_absolute():
        baseline_path = REPO_ROOT / baseline_path
    gate = None
    if baseline_path.is_file():
        baseline = np.load(baseline_path)
        if baseline.shape != scaffold.shape:
            baseline = resample_field(
                baseline, scaffold.shape[0], scaffold.shape[1])
        # AdaptivePixel starts coarse; load sites for the gate must match
        # the full-grid scaffold. PixelModel is already at that size.
        from neural_structural_optimization.models.model_pixel import PixelModel
        full_env = PixelModel(
            structural_params=golden.golden_structural_params(config),
            seed=config.seed,
        )
        nely = int(full_env.env.args['nely'])
        nelx = int(full_env.env.args['nelx'])
        sites = load_site_mask(
            full_env.env.args['forces'], nely=nely, nelx=nelx)
        gate = evaluate_tautology_gate(
            scaffold, baseline, sites, min_mass_off=args.min_mass_off)
        print(json.dumps({'gate': gate}, indent=2))
        if not gate['passed'] and not args.skip_gate:
            summary = {
                'prompt': config.prompt,
                'dream_steps': args.dream_steps,
                'dream_clip_loss': dream_losses[-1],
                'scaffold_threshold': threshold,
                'scaffold_mean': float(scaffold.mean()),
                'motif_scale_fracs': [],
                'union_load_sites': config.union_load_sites,
                'control_grid': [args.control_height, args.control_width],
                'dream_geometry': geometry,
                'gate': gate,
                'stopped': 'tautology_gate',
            }
            (output_dir / 'summary.json').write_text(
                json.dumps(summary, indent=2) + '\n')
            print(
                f'Tautology gate failed (spatial_mass_loss='
                f'{gate["spatial_mass_loss"]:.4f} < {args.min_mass_off}). '
                'Not running physics. Artifacts are in '
                f'{output_dir}.')
            return 2
    elif not args.skip_gate:
        raise FileNotFoundError(
            f'physics-layout proxy not found: {baseline_path}. '
            'Pass --baseline or --skip-gate.')

    if args.dream_only:
        summary = {
            'prompt': config.prompt,
            'dream_steps': args.dream_steps,
            'dream_clip_loss': dream_losses[-1],
            'scaffold_threshold': threshold,
            'scaffold_mean': float(scaffold.mean()),
            'motif_scale_fracs': [],
            'union_load_sites': config.union_load_sites,
            'control_grid': [args.control_height, args.control_width],
            'dream_geometry': geometry,
            'gate': gate,
            'stopped': 'dream_only',
        }
        (output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n')
        print(f'Dream-only complete. Artifacts in {output_dir}')
        return 0

    print('Physics pass: whole-building CLIP + Stage 6 occupancy anneal...')
    golden.seed_everything(config.seed)
    clip_loss = golden.build_clip_loss(config)
    golden.seed_everything(config.seed)
    physics_model = golden.build_model(clip_loss, config)
    apply_scaffold_as_occupancy_prior(
        physics_model,
        scaffold,
        weight=4000.0,
        weight_end=400.0,
        init_from_occupancy=True,
    )
    golden.seed_everything(config.seed)
    ds = golden.attach_final_raw_design(
        golden.build_optimizer(physics_model, config).optimize(),
        physics_model,
    )
    (output_dir / golden.REPLAY_FILENAME).write_text(
        json.dumps(golden.trajectory(ds), indent=2))

    density = np.asarray(ds['final_physical_density'].values, dtype=np.float32)
    replay_image, comparison_image = golden.save_motif_scale_look(
        ds,
        output_dir,
        replay_title=f'CLIP-dream layout, {config.prompt}',
        extra_panels=(
            ('Dream scaffold', output_dir / 'scaffold.png'),
        ),
    )
    nely = int(physics_model.env.args['nely'])
    nelx = int(physics_model.env.args['nelx'])
    sites = load_site_mask(
        physics_model.env.args['forces'], nely=nely, nelx=nelx)
    summary = {
        'prompt': config.prompt,
        'parameterization': 'adaptive_pixel',
        'dream_steps': args.dream_steps,
        'dream_clip_loss': dream_losses[-1],
        'scaffold_threshold': threshold,
        'scaffold_mean': float(scaffold.mean()),
        'motif_scale_fracs': [],
        'union_load_sites': config.union_load_sites,
        'control_grid': [args.control_height, args.control_width],
        'dream_geometry': geometry,
        'init_from_occupancy': True,
        'sketch_weight_start': 4000.0,
        'sketch_weight_end': 400.0,
        'gate': gate,
        'steps': int(ds.sizes['step']),
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': float(ds['clip_loss'][-1]),
        'volume_actual': golden.venice_volume_ratio(
            ds['final_design_raw'].values),
        'mean_physical_density': float(np.mean(density)),
        'mass_on_scaffold': mass_fraction_on_occupancy(density, scaffold),
        'spatial_mass_loss': scaffold_spatial_mass_loss(
            density, scaffold, sites),
        'paths': {
            'replay': str(replay_image),
            'comparison': str(comparison_image),
            'scaffold': str(output_dir / 'scaffold.png'),
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'\nWrote comparison to {comparison_image}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
