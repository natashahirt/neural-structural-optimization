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
from neural_structural_optimization.experiment import (
    VeniceGoldenConfig,
    _load_golden_script,
)
from neural_structural_optimization.models.loss_semantic_prior import (
    heaviside_projection,
    projected_density_view,
    report_design_metrics,
    save_design_arrays,
    sigma_for_min_feature,
)
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
DEFAULT_PROJECTION_SIGMA = 2.0
DEFAULT_PROJECTION_BETA_MAX = 8.0
# Quadratic volume penalty. CLIP loss on this path is O(0.4); a 2% mean
# miss then costs ~0.4 and a 5% miss dominates, so the term behaves like a
# constraint without a dual variable. Weaker weights let projection saturate.
DEFAULT_VOLUME_WEIGHT = 1000.0
DEFAULT_PHYSICS_PROXY = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / 'clip_motif_layout_no_occupancy' / 'teacher' / 'physical_density.npy')


def dream_results_dir(prompt: str, repo_root: Path = REPO_ROOT, golden=None) -> Path:
    golden = golden or _load_golden_script()
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


def _option_explicitly_passed(argv: Optional[list[str]], option: str) -> bool:
    tokens = sys.argv[1:] if argv is None else list(argv)
    prefix = option + '='
    return any(token == option or token.startswith(prefix) for token in tokens)


def _annealed_projection_beta(step: int, steps: int, beta_max: float) -> float:
    """Linear continuation from 1.0 to ``beta_max`` over ``steps`` iterates."""
    denom = max(int(steps) - 1, 1)
    return 1.0 + (float(beta_max) - 1.0) * (int(step) / denom)


def run_clip_dream(
    model,
    *,
    steps: int,
    lr: float,
    control_height: int = 16,
    control_width: int = 8,
    project_in_loop: bool = False,
    projection_sigma: float = DEFAULT_PROJECTION_SIGMA,
    projection_beta_max: float = DEFAULT_PROJECTION_BETA_MAX,
    dream_volume: Optional[float] = None,
    volume_weight: float = DEFAULT_VOLUME_WEIGHT,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    """Optimize a low-dimensional field decoded smoothly to the model grid.

    The 16x8 control has 128 variables instead of 2,048 independent pixels.
    Bilinear expansion plus one local average makes coherent regions cheap and
    removes the single-pixel texture move that dominated the direct dream.

    ``project_in_loop`` swaps the 3x3 box blur for filter-then-project
    (Gaussian + annealed Heaviside) and adds a quadratic volume penalty.
    Off by default so prior runs stay bit-identical. Returns
    ``(losses, decoded_field, control)``.
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
    field = None
    volume_target = (
        float(model.env.args['volfrac'])
        if dream_volume is None else float(dream_volume))
    for step in tqdm(range(int(steps)), desc='CLIP dream (no FEA)'):
        optimizer.zero_grad(set_to_none=True)
        logits = F.interpolate(
            control,
            size=model.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        if project_in_loop:
            # Density-like control, init at volfrac. A sigmoid centred at
            # eta=0.5 keeps gradients alive outside [0, 1]; a clamp would
            # freeze any pixel Adam pushed past the box and the volume term
            # could not pull it back. Gain 4 maps volfrac=0.3 to ~0.31, so
            # the starting mean is not shifted. Heaviside (inside
            # projected_density_view) then supplies binarity. Sigma is
            # cell-space, not sigma_for_min_feature's domain fraction.
            density = torch.sigmoid(4.0 * (logits - 0.5))[:, 0]
            beta = _annealed_projection_beta(
                step, steps, projection_beta_max)
            filter_sigma = float(projection_sigma)
            if filter_sigma <= 0.0:
                filter_sigma = sigma_for_min_feature(
                    int(density.shape[-1]), 0.0)
            # projected_density_view is gaussian_blur2d + heaviside_projection.
            projected = projected_density_view(
                density,
                beta=beta,
                eta=0.5,
                filter_sigma=filter_sigma,
            )
            volume_loss = (
                (projected.mean() - volume_target) ** 2
                * float(volume_weight))
            loss = model.get_semantic_loss(projected) + volume_loss
            field = projected
        else:
            logits = F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
            logits = logits[:, 0]
            loss = model.get_semantic_loss(logits)
            field = logits
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if field is None:
        raise RuntimeError('CLIP dream produced no field (steps must be > 0)')
    return losses, field.detach(), control.detach()


def extract_dream_scaffold(
    dream_field: np.ndarray,
    *,
    height: int,
    width: int,
    target_mean: float = DEFAULT_SCAFFOLD_ALLOWED_MEAN,
    passthrough: bool = False,
    soft_rank: bool = False,
):
    """Upsample the coarse dream and cut one elevation occupancy.

    Rank ink (logits are unbounded) and no distance envelope: the occupancy
    is the drawing, the way a Stage 6 sketch is one picture of the building.
    ``scale_fracs=(1.0,)`` is required by the extractor and unused at
    ``envelope_sigma_frac=0``.

    ``soft_rank`` converts unbounded dream logits to their continuous
    percentile ranks without cutting them. This preserves the grayscale
    organization CLIP optimized while making the field a valid ``[0, 1]``
    spatial preference map.

    ``passthrough`` skips the rank-cut: the in-loop projection already made a
    near-binary field at the dream volume, and recutting to
    ``target_allowed_mean`` would undo it. The upsample still runs so the
    scaffold matches the physics grid.
    """
    if passthrough and soft_rank:
        raise ValueError('passthrough and soft_rank are mutually exclusive')
    upsampled = resample_field(dream_field, height, width)
    if passthrough:
        return upsampled, None, np.asarray(upsampled)
    if soft_rank:
        return upsampled, None, rank_ink_from_raw(upsampled)
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


def _common_dream_summary(
    *,
    prompt: str,
    args,
    dream_losses: list[float],
    dream_coarse: np.ndarray,
    threshold,
    scaffold: np.ndarray,
    config,
    geometry: dict,
    gate,
    stopped: str,
    resolved_dream_volume: float,
) -> dict:
    """Shared dream keys so every exit path records the same contract."""
    return {
        'prompt': prompt,
        'dream_steps': int(args.dream_steps),
        'dream_clip_loss': float(dream_losses[-1]) if dream_losses else None,
        'dream_clip_losses': [float(x) for x in dream_losses],
        'dream_field_mean': float(np.mean(dream_coarse)),
        'scaffold_threshold': (
            None if threshold is None else float(threshold)),
        'scaffold_mean': float(np.mean(scaffold)),
        'scaffold_source': (
            'passthrough' if args.project_in_loop
            else 'soft_rank' if args.soft_scaffold
            else 'rank_cut'),
        'project_in_loop': bool(args.project_in_loop),
        'dream_volume': (
            float(resolved_dream_volume) if args.project_in_loop else None),
        'volume_weight': (
            float(args.volume_weight) if args.project_in_loop else None),
        'projection_sigma': (
            float(args.projection_sigma) if args.project_in_loop else None),
        'projection_beta_max': (
            float(args.projection_beta_max) if args.project_in_loop else None),
        'motif_scale_fracs': [],
        'union_load_sites': config.union_load_sites,
        'control_grid': [args.control_height, args.control_width],
        'dream_geometry': geometry,
        'gate': gate,
        'stopped': stopped,
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
        '--project-in-loop', action='store_true',
        help=(
            'filter-then-project inside the dream (Gaussian + annealed '
            'Heaviside) and skip the rank-cut scaffold. Off by default.'))
    parser.add_argument(
        '--soft-scaffold', action='store_true',
        help=(
            'preserve a continuous dream as a percentile-rank spatial '
            'preference map instead of thresholding it. Incompatible with '
            '--project-in-loop and explicit --target-allowed-mean.'))
    parser.add_argument(
        '--dream-field', type=Path,
        help=(
            'reuse an existing unbounded dream .npy instead of rerunning the '
            'CLIP-only stage; intended for controlled physics ablations.'))
    parser.add_argument(
        '--projection-sigma', type=float, default=DEFAULT_PROJECTION_SIGMA,
        help=(
            'Gaussian sigma in model-grid cells for --project-in-loop '
            f'(default {DEFAULT_PROJECTION_SIGMA:g}). Cell-space, not the '
            'domain-fraction form of sigma_for_min_feature.'))
    parser.add_argument(
        '--projection-beta-max', type=float,
        default=DEFAULT_PROJECTION_BETA_MAX,
        help=(
            'Heaviside beta at the last dream step, annealed from 1.0 '
            f'(default {DEFAULT_PROJECTION_BETA_MAX:g})'))
    parser.add_argument(
        '--dream-volume', type=float, default=None,
        help=(
            'target mean of the projected field. Default is the model '
            'volfrac. Only used with --project-in-loop.'))
    parser.add_argument(
        '--volume-weight', type=float, default=DEFAULT_VOLUME_WEIGHT,
        help=(
            'weight on (mean(rho) - dream_volume)^2 inside the dream '
            f'(default {DEFAULT_VOLUME_WEIGHT:g}). Constraint-like.'))
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
    if args.project_in_loop and _option_explicitly_passed(
            argv, '--target-allowed-mean'):
        raise ValueError(
            '--target-allowed-mean cannot be combined with --project-in-loop: '
            'the projected field is already at the dream volume and is not '
            'rank-cut. Drop one of the two flags.')
    if args.soft_scaffold and args.project_in_loop:
        raise ValueError(
            '--soft-scaffold cannot be combined with --project-in-loop')
    if args.soft_scaffold and _option_explicitly_passed(
            argv, '--target-allowed-mean'):
        raise ValueError(
            '--target-allowed-mean cannot be combined with --soft-scaffold: '
            'the continuous rank field is not cut.')

    golden = _load_golden_script()
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
    model = build_dream_model(config, golden)
    resolved_dream_volume = (
        float(model.env.args['volfrac'])
        if args.dream_volume is None else float(args.dream_volume))
    if args.dream_field is not None:
        dream_path = args.dream_field
        if not dream_path.is_absolute():
            dream_path = REPO_ROOT / dream_path
        print(f'Reusing CLIP dream field from {dream_path}; no dream optimization.')
        dream_coarse = np.asarray(np.load(dream_path), dtype=np.float32)
        while dream_coarse.ndim > 2:
            dream_coarse = dream_coarse[0]
        if dream_coarse.ndim != 2:
            raise ValueError(
                f'--dream-field must resolve to a 2-D field, got '
                f'{dream_coarse.shape}')
        dream_losses = []
        control_field = None
    else:
        print(
            f'CLIP dream of the whole elevation, prompt={config.prompt!r}, '
            f'{args.dream_steps} steps, lr={args.dream_lr:g}, no FEA, '
            f'{args.control_height}x{args.control_width} smooth control, '
            'no motif-scale crops, no load-site frames.')
        dream_losses, dream_logits, dream_control = run_clip_dream(
            model,
            steps=args.dream_steps,
            lr=args.dream_lr,
            control_height=args.control_height,
            control_width=args.control_width,
            project_in_loop=args.project_in_loop,
            projection_sigma=args.projection_sigma,
            projection_beta_max=args.projection_beta_max,
            dream_volume=resolved_dream_volume,
            volume_weight=args.volume_weight,
        )
        dream_coarse = np.asarray(
            dream_logits.cpu().numpy(), dtype=np.float32)
        while dream_coarse.ndim > 2:
            dream_coarse = dream_coarse[0]
        control_field = np.asarray(
            dream_control.detach().cpu().numpy(), dtype=np.float32)
        while control_field.ndim > 2:
            control_field = control_field[0]
    np.save(output_dir / 'dream_coarse.npy', dream_coarse)
    if control_field is not None:
        np.save(output_dir / 'control.npy', control_field)
    rank_preview = not args.project_in_loop
    _save_field_png(
        output_dir / 'dream_coarse.png', dream_coarse, rank=rank_preview)

    upsampled, threshold, scaffold = extract_dream_scaffold(
        dream_coarse,
        height=config.height,
        width=config.width,
        target_mean=args.target_allowed_mean,
        passthrough=args.project_in_loop,
        soft_rank=args.soft_scaffold,
    )
    np.save(output_dir / 'dream_upsampled.npy', upsampled)
    np.save(output_dir / 'scaffold.npy', scaffold)
    _save_field_png(
        output_dir / 'dream_upsampled.png', upsampled, rank=rank_preview)
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
            summary = _common_dream_summary(
                prompt=config.prompt,
                args=args,
                dream_losses=dream_losses,
                dream_coarse=dream_coarse,
                threshold=threshold,
                scaffold=scaffold,
                config=config,
                geometry=geometry,
                gate=gate,
                stopped='tautology_gate',
                resolved_dream_volume=resolved_dream_volume,
            )
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
        summary = _common_dream_summary(
            prompt=config.prompt,
            args=args,
            dream_losses=dream_losses,
            dream_coarse=dream_coarse,
            threshold=threshold,
            scaffold=scaffold,
            config=config,
            geometry=geometry,
            gate=gate,
            stopped='dream_only',
            resolved_dream_volume=resolved_dream_volume,
        )
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
    array_paths = save_design_arrays(
        output_dir, density, raw=ds['final_design_raw'].values)
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
    report = report_design_metrics(
        density, sites, scaffold=scaffold, ds=ds)
    summary = _common_dream_summary(
        prompt=config.prompt,
        args=args,
        dream_losses=dream_losses,
        dream_coarse=dream_coarse,
        threshold=threshold,
        scaffold=scaffold,
        config=config,
        geometry=geometry,
        gate=gate,
        stopped='physics',
        resolved_dream_volume=resolved_dream_volume,
    )
    summary.update({
        'parameterization': 'adaptive_pixel',
        'init_from_occupancy': True,
        'sketch_weight_start': 4000.0,
        'sketch_weight_end': 400.0,
        'steps': int(ds.sizes['step']),
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': report['clip_loss'],
        'clip_loss_raw': report['clip_loss_raw'],
        'volume_actual': golden.venice_volume_ratio(
            ds['final_design_raw'].values),
        'mean_physical_density': report['mean_physical_density'],
        'mass_on_scaffold': report['mass_on_scaffold'],
        'spatial_mass_loss': report['spatial_mass_loss'],
        'validity': report['validity'],
        'paths': {
            'replay': str(replay_image),
            'comparison': str(comparison_image),
            'scaffold': str(output_dir / 'scaffold.png'),
            'density_npy': str(array_paths['physical_density']),
            'raw_npy': str(array_paths['final_design_raw']),
        },
    })
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'\nWrote comparison to {comparison_image}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
