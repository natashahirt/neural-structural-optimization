"""Coarse CLIP/SDS spatial-prior experiment against exact FEA compliance.

Matched arms on a 32x64 (width x height) four-storey building:

1. compliance only
2. Venice scalar CLIP coupling
3. scalar CLIP plus a live CLIP occupancy prior
4. hierarchical (global -> storey -> member) CLIP prior
5. frozen SDS prior through the same occupancy interface

No generated reference image. Prior weights are set from the measured
guidance-to-compliance gradient-norm ratio.
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
from neural_structural_optimization.experiment import VeniceGoldenConfig
from neural_structural_optimization.models.loss_semantic_prior import (
    CLIPSemanticProvider,
    DiffusionSDSProvider,
    FrozenDenoiser,
    SemanticSpatialPrior,
    connectivity_metrics,
    scale_fracs_for_grid,
)
from neural_structural_optimization.models.loss_sketch import load_site_mask

import importlib.util


def _load_golden():
    spec = importlib.util.spec_from_file_location(
        'venice_golden_250214', REPO_ROOT / 'script' / 'venice_golden_250214.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden = _load_golden()

COARSE_WIDTH = 32
COARSE_HEIGHT = 64
COARSE_INTERVAL = 16
DEFAULT_STEPS = 60
DEFAULT_AUTO_RATIO = 0.25
FULL_WIDTH = 128
FULL_HEIGHT = 256
FULL_INTERVAL = 64


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


def _save_field(path: Path, field: np.ndarray) -> Path:
    arr = np.asarray(field, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    Image.fromarray((255.0 * (1.0 - norm)).clip(0, 255).astype(np.uint8), mode='L').save(path)
    return path


def _stack_panels(path: Path, panels: list[tuple[str, np.ndarray]]) -> Path:
    from PIL import ImageDraw

    images = []
    for title, field in panels:
        arr = np.asarray(field, dtype=np.float64)
        while arr.ndim > 2:
            arr = arr[0]
        lo, hi = float(arr.min()), float(arr.max())
        norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        images.append((title, Image.fromarray(
            (255.0 * (1.0 - norm)).clip(0, 255).astype(np.uint8), mode='L')))
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


def _results_dir(stem: str) -> Path:
    path = REPO_ROOT / 'script' / 'resources' / 'results' / stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attach_prior(model, config: VeniceGoldenConfig, *, kind: str, curriculum: str):
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
    )
    model.enable_semantic_prior(prior)
    return prior


def _validity(model) -> dict:
    density = model.get_physical_density(model.z).detach().cpu().numpy()
    while density.ndim > 2:
        density = density[0]
    nely = int(model.env.args['nely'])
    nelx = int(model.env.args['nelx'])
    sites = load_site_mask(model.env.args['forces'], nely=nely, nelx=nelx)
    metrics = connectivity_metrics(density, sites, threshold=0.3)
    metrics['mean_physical_density'] = float(density.mean())
    return metrics, density, sites


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
        _attach_prior(model, config, kind=prior_kind, curriculum=curriculum)
    ds = golden.build_optimizer(model, config).optimize()
    ds = golden.attach_final_raw_design(ds, model)
    validity, density, sites = _validity(model)
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
    _save_field(out / 'physical_density.png', density)
    _save_field(out / 'compliance_sensitivity.png', sensitivity)
    panels = [('density', density), ('dC/dz', sensitivity)]
    if occupancy is not None:
        _save_field(out / 'semantic_occupancy.png', occupancy)
        panels.append(('CLIP/SDS prior', occupancy))
    if preference is not None:
        _save_field(out / 'semantic_preference.png', preference)
        panels.append(('preference', preference))
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
        'clip_loss': float(ds['clip_loss'][-1]) if 'clip_loss' in ds else None,
        'clip_loss_raw': float(ds['clip_loss_raw'][-1]) if 'clip_loss_raw' in ds else None,
        'volume_actual': golden.venice_volume_ratio(ds['final_design_raw'].values),
        'prior_weight': None if prior is None else float(prior.weight),
        'semantic_metrics_final': None if prior is None else dict(prior.last_metrics),
        'validity': validity,
        'paths': {
            'comparison': str(out / 'comparison.png'),
            'density': str(out / 'physical_density.png'),
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
        default='compliance,scalar_clip,clip_prior,hierarchical,venation,sds_prior',
        help='Comma-separated arm names to run.',
    )
    args = parser.parse_args(argv)
    wanted = {name.strip() for name in args.arms.split(',') if name.strip()}
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

    if 'venation' in wanted:
        venation = coarse_config('butterfly wing venation', max_iterations=args.steps)
        summaries['venation'] = run_arm(
            'butterfly_wing_venation_clip_prior', 'butterfly wing venation',
            with_clip=True, prior_kind='clip', curriculum='global_only',
            config=venation)

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
        full = VeniceGoldenConfig(
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
        summaries['full_replay'] = run_arm(
            f'{golden.prompt_slug(prompt)}_full_replay', prompt,
            with_clip=True, prior_kind='clip', curriculum='hierarchical',
            config=full)

    index = _results_dir('clip_saliency_compliance_index')
    (index / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
