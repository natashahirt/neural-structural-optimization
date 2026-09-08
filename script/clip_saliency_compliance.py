"""Coarse CLIP/SDS spatial-prior experiment against exact FEA compliance.

Matched arms on a 32x64 (width x height) four-storey building:

1. compliance only
2. Venice scalar CLIP coupling
3. scalar CLIP plus a live CLIP occupancy prior
4. hierarchical (global -> storey -> member) CLIP prior
5. hierarchical prior whose sculptural scale is scored on a
   filter-then-project view, so CLIP cannot draw with near-void gray
6. frozen SDS prior through the same occupancy interface (opt-in via
   ``--arms sds_prior``; not in the default arm list)

No generated reference image. Prior weights are set from the measured
guidance-to-compliance gradient-norm ratio.

A "full replay" is not a weight reload. Coarse arms (32x64) are the
causal test; if a prior arm stays connected and beats scalar CLIP on
``clip_loss_raw``, the same mechanism is run again from scratch at
128x256 so the motif is actually readable. Outputs live under
``script/resources/results/clip_saliency_compliance/``.
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import configure_torch_threads
from neural_structural_optimization.experiment import (
    VeniceGoldenConfig,
    _load_golden_script,
)
from neural_structural_optimization.models.loss_semantic_prior import (
    CLIPSemanticProvider,
    DiffusionSDSProvider,
    FrozenDenoiser,
    SemanticSpatialPrior,
    physical_density_and_sites,
    report_design_metrics,
    save_design_arrays,
    scale_fracs_for_grid,
)

golden = _load_golden_script()

COARSE_WIDTH = 32
COARSE_HEIGHT = 64
COARSE_INTERVAL = 16
DEFAULT_STEPS = 60
DEFAULT_AUTO_RATIO = 0.25
# Sculptural projection: sharp enough to erase the faint tail without
# flattening the mid-density gradient the prior still needs at neutral init.
DEFAULT_PROJECTION_BETA = 8.0
DEFAULT_PROJECTION_SIGMA = 2.0
FULL_WIDTH = 128
FULL_HEIGHT = 256
FULL_INTERVAL = 64
DEFAULT_ARMS = (
    'compliance,scalar_clip,clip_prior,hierarchical,projected')
FOLDED_ARM_NAMES = {
    'venation': (
        'use --prompt "butterfly wing venation" --arms clip_prior'),
    'venation_projected': (
        'use --prompt "butterfly wing venation" --arms projected'),
}


def coarse_config(prompt: str, **overrides) -> VeniceGoldenConfig:
    values = dict(
        width=COARSE_WIDTH,
        height=COARSE_HEIGHT,
        interval=COARSE_INTERVAL,
        density=0.3,
        filter_width=2.0,
        penal=3.0,
        resize_num=0,
        resize_scale=2,
        prompt=prompt,
        num_augs=8,
        clip_alpha=10.0,
        lr=0.2,
        max_iterations=DEFAULT_STEPS,
        max_resize_iteration=50,
        convergence_threshold=0.05,
        seed=12,
        device='cpu',
        motif_scale_fracs=(),
        motif_scale_crops=4,
        motif_scale_weight=1.0,
        neutral_init=True,
        init_noise_amp=0.01,
        union_load_sites=True,
    )
    values.update(overrides)
    return VeniceGoldenConfig(**values)


def full_grid_config(prompt: str) -> VeniceGoldenConfig:
    """128x256 replay grid shared by projected_full and auto full_replay."""
    return VeniceGoldenConfig(
        width=FULL_WIDTH,
        height=FULL_HEIGHT,
        interval=FULL_INTERVAL,
        density=0.3,
        resize_num=2,
        prompt=prompt,
        num_augs=8,
        clip_alpha=10.0,
        lr=0.2,
        max_iterations=80,
        max_resize_iteration=30,
        seed=12,
        motif_scale_fracs=(),
        neutral_init=True,
    )


def _squeeze2d(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _field_to_uint8(field: np.ndarray, *, scale: str) -> np.ndarray:
    """Render a field as inverted 8-bit grayscale.

    ``absolute`` maps physical [0, 1] onto the full gray ramp so two runs
    are comparable. ``normalized`` min-max stretches the array; use it only
    when the field has no meaningful absolute scale.
    """
    arr = _squeeze2d(field)
    if scale == 'absolute':
        norm = np.clip(arr, 0.0, 1.0)
    elif scale == 'normalized':
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            norm = (arr - lo) / (hi - lo)
        else:
            norm = np.zeros_like(arr)
    else:
        raise ValueError(
            f'unknown field scale {scale!r}; expected absolute or normalized')
    return (255.0 * (1.0 - norm)).clip(0, 255).astype(np.uint8)


def _save_field(path: Path, field: np.ndarray, *, scale: str) -> Path:
    Image.fromarray(_field_to_uint8(field, scale=scale), mode='L').save(path)
    return path


def _stack_panels(
    path: Path,
    panels: list[tuple[str, np.ndarray, str]],
) -> Path:
    from PIL import ImageDraw

    images = [
        (title, Image.fromarray(_field_to_uint8(field, scale=scale), mode='L'))
        for title, field, scale in panels
    ]
    gap, label_h = 8, 18
    width, height = images[0][1].size
    canvas = Image.new('L', (width * len(images) + gap * (len(images) - 1), height + label_h), 255)
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(images):
        x = i * (width + gap)
        canvas.paste(image, (x, label_h))
        draw.text((x + 4, 2), title, fill=0)
    canvas.save(path)
    return path


FAMILY_DIR = REPO_ROOT / 'script' / 'resources' / 'results' / 'clip_saliency_compliance'


def _results_dir(stem: str) -> Path:
    """Arm directory under the saliency-compliance family.

    ``stem`` may be a bare arm name (``human_skull_projected``) or the
    legacy ``clip_saliency_compliance_<arm>`` prefix.
    """
    name = stem.removeprefix('clip_saliency_compliance_')
    path = FAMILY_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attach_prior(
    model,
    config: VeniceGoldenConfig,
    *,
    kind: str,
    curriculum: str,
    projection_beta: float = 0.0,
    projection_filter_sigma: float = 0.0,
    projection_scales: tuple[str, ...] = ('global',),
):
    interval = int(model.env.args.get('interval', config.interval))
    height = int(model.full_params.height) if hasattr(model, 'full_params') else config.height
    fracs = scale_fracs_for_grid(height, interval)
    if kind == 'clip':
        provider = CLIPSemanticProvider(model.clip_loss, fracs)
    elif kind == 'sds':
        cond = None
        clip = model.clip_loss
        if clip is not None and getattr(clip, 'e_pos', None) is not None:
            cond = clip.e_pos.detach()
        provider = DiffusionSDSProvider(
            FrozenDenoiser(channels=16, cond_dim=(0 if cond is None else int(cond.numel())), seed=config.seed),
            cond=cond,
            seed=config.seed,
        )
    else:
        raise ValueError(f'unknown semantic provider {kind!r}')
    prior = SemanticSpatialPrior(
        provider,
        scale_fracs=fracs,
        weight=0.0,
        auto_ratio=DEFAULT_AUTO_RATIO,
        ema_decay=0.9,
        smooth_sigma=1.5,
        curriculum=curriculum,
        record_alignment=True,
        projection_beta=projection_beta,
        projection_filter_sigma=projection_filter_sigma,
        projection_scales=projection_scales,
    )
    model.enable_semantic_prior(prior)
    return prior


def _compliance_sensitivity(model) -> np.ndarray:
    logits = model.z
    model.zero_grad(set_to_none=True)
    compliance = model.get_structural_loss(logits)
    grad, = torch.autograd.grad(compliance, logits, retain_graph=False)
    field = grad.detach().cpu().numpy()
    while field.ndim > 2:
        field = field[0]
    return field


def run_arm(
    name: str,
    prompt: str,
    *,
    with_clip: bool,
    prior_kind: Optional[str],
    curriculum: str = 'global_only',
    config: Optional[VeniceGoldenConfig] = None,
    projection_beta: float = 0.0,
    projection_filter_sigma: float = 0.0,
    projection_scales: tuple[str, ...] = ('global',),
) -> dict:
    config = config or coarse_config(prompt)
    out = _results_dir(f'clip_saliency_compliance_{name}')
    configure_torch_threads()
    clip_loss = golden.build_clip_loss(config) if with_clip else None
    golden.seed_everything(config.seed)
    model = golden.build_model(clip_loss, config)
    golden.seed_everything(config.seed)
    if prior_kind is not None:
        if clip_loss is None:
            raise ValueError('a semantic prior requires CLIP weights for the text embed / CLIP provider')
        _attach_prior(
            model, config, kind=prior_kind, curriculum=curriculum,
            projection_beta=projection_beta,
            projection_filter_sigma=projection_filter_sigma,
            projection_scales=projection_scales)
    ds = golden.build_optimizer(model, config).optimize()
    ds = golden.attach_final_raw_design(ds, model)
    density, sites = physical_density_and_sites(model)
    sensitivity = _compliance_sensitivity(model)
    prior = model.semantic_prior
    occupancy = None
    preference = None
    if prior is not None and prior.maps.get('global') is not None:
        occupancy = prior.blended_occupancy(
            torch.as_tensor(density)).detach().cpu().numpy()
        while occupancy.ndim > 2:
            occupancy = occupancy[0]
        preference = prior.last_clip_grad
        if preference is not None:
            preference = preference.detach().cpu().numpy()
            while preference.ndim > 2:
                preference = preference[0]
    report = report_design_metrics(
        density, sites, ds=ds, scaffold=occupancy)
    array_paths = save_design_arrays(
        out, density, raw=ds['final_design_raw'].values)
    _save_field(out / 'physical_density.png', density, scale='absolute')
    _save_field(out / 'compliance_sensitivity.png', sensitivity, scale='normalized')
    panels = [
        ('density', density, 'absolute'),
        ('dC/dz', sensitivity, 'normalized'),
    ]
    if occupancy is not None:
        _save_field(out / 'semantic_occupancy.png', occupancy, scale='absolute')
        panels.append(('CLIP/SDS prior', occupancy, 'absolute'))
    if preference is not None:
        _save_field(out / 'semantic_preference.png', preference, scale='normalized')
        panels.append(('preference', preference, 'normalized'))
    _stack_panels(out / 'comparison.png', panels)

    traj = golden.trajectory(ds)
    (out / golden.REPLAY_FILENAME).write_text(json.dumps(traj, indent=2))
    summary = {
        'arm': name,
        'prompt': prompt,
        'prior_kind': prior_kind,
        'curriculum': curriculum if prior_kind else None,
        'grid': [int(density.shape[0]), int(density.shape[1])],
        'steps': int(ds.sizes['step']),
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': report['clip_loss'],
        'clip_loss_raw': report['clip_loss_raw'],
        'volume_actual': golden.venice_volume_ratio(ds['final_design_raw'].values),
        'mean_physical_density': report['mean_physical_density'],
        'mass_on_scaffold': report['mass_on_scaffold'],
        'spatial_mass_loss': report['spatial_mass_loss'],
        'prior_weight': None if prior is None else float(prior.weight),
        'projection': None if prior is None else {
            'beta': float(prior.projection_beta),
            'eta': float(prior.projection_eta),
            'filter_sigma': float(prior.projection_filter_sigma),
            'scales': list(prior.projection_scales),
        },
        'semantic_metrics_final': None if prior is None else dict(prior.last_metrics),
        'validity': report['validity'],
        'paths': {
            'comparison': str(out / 'comparison.png'),
            'density': str(out / 'physical_density.png'),
            'density_npy': str(array_paths['physical_density']),
            'raw_npy': str(array_paths['final_design_raw']),
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def _should_replay_full(prior_summary: dict, scalar_summary: dict) -> bool:
    if not prior_summary['validity']['support_to_load_connected']:
        return False
    prior_clip = prior_summary.get('clip_loss_raw')
    scalar_clip = scalar_summary.get('clip_loss_raw')
    if prior_clip is None or scalar_clip is None:
        return False
    return float(prior_clip) < float(scalar_clip)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='human skull')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--skip-full', action='store_true')
    parser.add_argument(
        '--arms',
        default=DEFAULT_ARMS,
        help=(
            'Comma-separated arm names. Default omits sds_prior (a '
            'random-weight FrozenDenoiser control, not a diffusion prior); '
            'pass --arms sds_prior to run it. Venation is --prompt, not an '
            'arm name.'),
    )
    parser.add_argument(
        '--projection-beta', type=float, default=DEFAULT_PROJECTION_BETA,
        help='Heaviside sharpness for the sculptural arm. 0 disables projection.',
    )
    parser.add_argument(
        '--projection-sigma', type=float, default=DEFAULT_PROJECTION_SIGMA,
        help='Blur sigma in cells, imposing a minimum semantic feature size.',
    )
    parser.add_argument(
        '--projection-scales', default='global',
        help='Comma-separated scales scored on the projected view.',
    )
    args = parser.parse_args(argv)
    projection_scales = tuple(
        name.strip() for name in args.projection_scales.split(',') if name.strip())
    wanted = {name.strip() for name in args.arms.split(',') if name.strip()}
    folded = wanted & FOLDED_ARM_NAMES.keys()
    if folded:
        hints = '; '.join(
            f'{name}: {FOLDED_ARM_NAMES[name]}' for name in sorted(folded))
        raise ValueError(f'folded arm names: {hints}')
    prompt = args.prompt
    config = coarse_config(prompt, max_iterations=args.steps)
    summaries = {}

    if 'compliance' in wanted:
        summaries['compliance'] = run_arm(
            f'{golden.prompt_slug(prompt)}_compliance', prompt,
            with_clip=False, prior_kind=None, config=config)

    if 'scalar_clip' in wanted:
        summaries['scalar_clip'] = run_arm(
            f'{golden.prompt_slug(prompt)}_scalar_clip', prompt,
            with_clip=True, prior_kind=None, config=config)

    if 'clip_prior' in wanted:
        summaries['clip_prior'] = run_arm(
            f'{golden.prompt_slug(prompt)}_clip_prior', prompt,
            with_clip=True, prior_kind='clip', curriculum='global_only',
            config=config)

    if 'hierarchical' in wanted:
        hier = dataclasses.replace(
            config, resize_num=1, max_resize_iteration=max(args.steps // 3, 8),
            max_iterations=args.steps)
        summaries['hierarchical'] = run_arm(
            f'{golden.prompt_slug(prompt)}_hierarchical', prompt,
            with_clip=True, prior_kind='clip', curriculum='hierarchical',
            config=hier)

    if 'projected' in wanted:
        sculpt = dataclasses.replace(
            config, resize_num=1, max_resize_iteration=max(args.steps // 3, 8),
            max_iterations=args.steps)
        summaries['projected'] = run_arm(
            f'{golden.prompt_slug(prompt)}_projected', prompt,
            with_clip=True, prior_kind='clip', curriculum='hierarchical',
            projection_beta=args.projection_beta,
            projection_filter_sigma=args.projection_sigma,
            projection_scales=projection_scales,
            config=sculpt)

    if 'compliance_full' in wanted:
        summaries['compliance_full'] = run_arm(
            f'{golden.prompt_slug(prompt)}_compliance_full', prompt,
            with_clip=False, prior_kind=None,
            config=full_grid_config(prompt))

    if 'projected_full' in wanted:
        summaries['projected_full'] = run_arm(
            f'{golden.prompt_slug(prompt)}_projected_full', prompt,
            with_clip=True, prior_kind='clip', curriculum='hierarchical',
            projection_beta=args.projection_beta,
            projection_filter_sigma=args.projection_sigma,
            projection_scales=projection_scales,
            config=full_grid_config(prompt))

    if 'sds_prior' in wanted:
        summaries['sds_prior'] = run_arm(
            f'{golden.prompt_slug(prompt)}_sds_prior', prompt,
            with_clip=True, prior_kind='sds', curriculum='global_only',
            config=config)

    if (
        not args.skip_full
        and 'clip_prior' in summaries
        and 'scalar_clip' in summaries
        and _should_replay_full(summaries['clip_prior'], summaries['scalar_clip'])
    ):
        summaries['full_replay'] = run_arm(
            f'{golden.prompt_slug(prompt)}_full_replay', prompt,
            with_clip=True, prior_kind='clip', curriculum='hierarchical',
            config=full_grid_config(prompt))

    FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    (FAMILY_DIR / 'index.json').write_text(json.dumps(summaries, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
