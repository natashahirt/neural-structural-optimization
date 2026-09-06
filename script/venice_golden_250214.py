"""The Venice ``250214_skeleton_loss_test_balanced_dynamic`` golden run.

This is the reference configuration the hardfork has to reproduce before any
further architectural change is worth trusting: the legacy Venice repository's
known-good "skeletons" run, expressed against this repository's Venice
compatibility preset. Every value below is taken from the run's own log
(``results/250214_skeleton_loss_test_balanced_dynamic/2_final/00-log-*.json``)
rather than from Venice's checked-in ``main.py``, which has since drifted.

Run it with::

    PYTHONPATH="$PWD" python script/venice_golden_250214.py

which takes roughly a quarter of an hour on CPU. The builders are importable
individually so that `tests/test_venice_parity.py` drives the same
configuration without restating it.

Three details decide whether the trajectory reproduces, and each is easy to get
wrong:

* The design is seeded from an IMAGE, in [0, 1] pixel space, with no logit.
* The image is loaded at the schedule's COARSE resolution (32x64), not at the
  final 128x256, and reaches the final grid only through the upsamples.
* The learning rate is 0.2. `AdaptiveAdam_Optimizer` defaults to this
  repository's 1e-2, so it has to be passed explicitly; every other default in
  the model and the optimizer is already Venice's.
"""

import argparse
import dataclasses
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import configure_torch_threads
from neural_structural_optimization.experiment import (
    GOLDEN,
    SMOKE,
    VeniceGoldenConfig,
    venice_250214_motif_scale,
)
from neural_structural_optimization.models import AdaptivePixelModel, CLIPLoss
from neural_structural_optimization.models.loss_clip import VeniceClipPreset
from neural_structural_optimization.models.model_base import VeniceLossAlgebra
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.train import AdaptiveAdam_Optimizer
from neural_structural_optimization.train.utils import (
    init_weight_neutral,
    init_weight_with_image,
)

# Copied out of the Venice tree (commit 97f9336) rather than referenced across
# repositories. A golden run that reads its initial condition from a sibling
# checkout is not reproducible: it silently changes meaning when that checkout
# moves, and fails outright when it is absent. The relative path under
# `resources/` is preserved so the provenance stays obvious, and the file is
# byte-identical (sha256 750bb809f38c0f5ef6f9af51f228a860a40287b56efde07e27dc061eff157d98).
GOLDEN_IMAGE_PATH = (
    REPO_ROOT / 'script' / 'resources' / 'input_images' / 'dSketches'
    / 'thick_outer_lins.png')

# Venice's own final design image, copied out of the same commit for the same
# reason. The golden JSON records only scalars, so this is the reference run's
# only surviving record of what it actually built, and it is what the design
# comparison in `tests/test_venice_parity.py` is pinned against. It is the
# design parameter resized to `img_width` and then clamped and inverted -- see
# `attach_final_raw_design`.
GOLDEN_FINAL_IMAGE_PATH = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / '250214_skeleton_loss_test_balanced_dynamic' / '2_final'
    / ('0_final-P_multistory_building-M_Ada-T_skeletons-W_128-H_256-V_0.30'
       '-LR_0.20-CW_10-ID_01-CompL_74.00-ClipL_0.38-VA_0.31'
       '-balanced_dynamic.jpg'))


# Knobs live on `neural_structural_optimization.experiment.VeniceGoldenConfig`
# so `--print-config` can inspect them without importing this script's CLIP
# builders. GOLDEN / SMOKE / VeniceGoldenConfig are re-exported here so the
# parity harness and this runner keep one import path.

REPLAY_FILENAME = 'venice_golden_250214_replay.json'

# What the reference run produced, for reporting at the end of a replay. The
# parity harness holds the full trajectory; these are its last recorded point.
GOLDEN_FINAL_ITERATION = 124
GOLDEN_FINAL = {
    'compliance': 73.99691670938864,
    'clip_loss': 283.3703603894398,
    'clip_loss_raw': 0.3829488754272461,
    'clip_weight': 739.9691670938864,
    'total': 357.75022597425567,
    # params['volume_actual'], i.e. `venice_volume_ratio` of the final design.
    'volume_actual': 0.3148193359375,
}


def seed_everything(seed: int) -> None:
    """Seed torch, numpy and python `random` before the optimization loop.

    Venice seeds only torch and numpy (`models.py` line 742), but the
    augmentation stack can reach python's `random` through its dependencies, so
    it is pinned too -- a strictly tighter constraint than the reference run's.

    This makes a replay reproducible WITHIN this repository. It does not make
    it bit-identical to Venice: Kornia draws from torch's global generator, so
    the crop sequence depends on how many draws everything upstream consumed,
    and that count differs between the two codebases. See the tolerance
    discussion in `tests/test_venice_parity.py`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def golden_structural_params(config: VeniceGoldenConfig = GOLDEN) -> StructuralParams:
    """Build the structural problem at its final resolution."""
    return StructuralParams(
        problem_name=config.problem_name,
        width=config.width,
        height=config.height,
        density=config.density,
        interval=config.interval,
        filter_width=config.filter_width,
    )


def venice_clip_preset(config: VeniceGoldenConfig = GOLDEN) -> VeniceClipPreset:
    """Build the CLIP preset for `config`.

    For `GOLDEN` this is `VENICE_CLIP_PRESET` unchanged -- the module's defaults
    are already the reference run's -- and only the crop count and view size are
    parameterized, so a cheaper configuration can reuse the same path.
    """
    return VeniceClipPreset(
        resize_short_side=config.clip_resize_short_side,
        num_augs=config.num_augs,
    )


def build_clip_loss(config: VeniceGoldenConfig = GOLDEN) -> CLIPLoss:
    """Build the CLIP loss on the Venice-compatible path.

    The preset carries the rest of the reference settings: the 480px
    `RandomResizedCrop` reference size, 0.1 augmentation noise and the arcsine
    distance transform.
    """
    return CLIPLoss(
        clip_model_name=config.clip_model_name,
        clip_rn_model_name=config.clip_rn_model_name,
        device=torch.device(config.device),
        positive_prompts=[config.prompt],
        num_augs=config.num_augs,
        venice_compat=venice_clip_preset(config),
        motif_scale_fracs=config.motif_scale_fracs,
        motif_scale_crops=config.motif_scale_crops,
        motif_scale_weight=config.motif_scale_weight,
    )


def build_model(
    clip_loss: Optional[CLIPLoss] = None,
    config: VeniceGoldenConfig = GOLDEN,
    image_path: Optional[Path] = None,
) -> AdaptivePixelModel:
    """Build the adaptive model, coarse-started and seeded.

    The frozen Venice path seeds from an image. Motif research defaults
    (``neutral_init``) use uniform volume fraction plus tiny noise instead.

    Args:
        clip_loss: the semantic loss, or None for a structure-only replay.
        config: the run configuration.
        image_path: override for the initial image; defaults to the copied
            Venice asset. Ignored when ``config.neutral_init`` is True.

    Returns:
        A model at the schedule's coarse resolution whose design parameter is
        the seed field, with the legacy loss algebra enabled.
    """
    model = AdaptivePixelModel(
        structural_params=golden_structural_params(config),
        clip_loss=clip_loss,
        seed=config.seed,
        resize_num=config.resize_num,
        resize_scale=config.resize_scale,
    )
    if model.args['penal'] != config.penal:
        raise ValueError(
            f"physics penal is {model.args['penal']}, not the configured "
            f'{config.penal}; the reference run cannot be reproduced.')

    # `PixelModel`/`CNNModel` do not forward `venice_compat` to `Model`, so the
    # seam is flipped after construction for every model alike.
    model.enable_venice_compat_loss(VeniceLossAlgebra(
        clip_alpha=config.clip_alpha,
        compliance_weight=config.compliance_weight,
    ))

    if config.neutral_init:
        init_weight_neutral(
            model,
            density=config.density,
            seed=config.seed,
            noise_amp=config.init_noise_amp,
            union_load_sites=config.union_load_sites,
        )
    else:
        init_weight_with_image(
            model,
            GOLDEN_IMAGE_PATH if image_path is None else image_path,
            invert_image=config.invert_image,
        )
    return model


def build_optimizer(
    model: AdaptivePixelModel,
    config: VeniceGoldenConfig = GOLDEN,
    max_iterations: Optional[int] = None,
) -> AdaptiveAdam_Optimizer:
    """Build the optimizer driving the resolution schedule.

    `clip_alpha` is deliberately NOT passed: the model's algebra already owns
    it, and passing it here would override the algebra with the same number by
    a different route.
    """
    return AdaptiveAdam_Optimizer(
        model,
        max_iterations=(config.max_iterations if max_iterations is None
                        else max_iterations),
        lr=config.lr,
        compliance_weight=config.compliance_weight,
        resize_threshold=config.resize_threshold,
        max_resize_iteration=config.max_resize_iteration,
        convergence_threshold=config.convergence_threshold,
    )


def run(
    config: VeniceGoldenConfig = GOLDEN,
    max_iterations: Optional[int] = None,
    with_clip: bool = True,
) -> xarray.Dataset:
    """Replay the golden run and return its dataset.

    The returned dataset carries the per-step term breakdown (`compliance`,
    `clip_loss`, `clip_loss_raw`, `clip_weight`) beside the scalar `loss`, so a
    replay can be compared against the reference log field by field. It also
    carries `final_design_raw`, the design parameter as the run left it -- see
    :func:`attach_final_raw_design`.

    Args:
        config: the run configuration.
        max_iterations: cap on gradient steps, for a shortened replay.
        with_clip: build the CLIP loss. False gives a structure-only run, which
            is useful for exercising the schedule without the CLIP cost.
    """
    # The package pins OMP_NUM_THREADS=1 at import to keep CHOLMOD from racing,
    # which would otherwise leave torch single-threaded and roughly double the
    # CLIP forward. Restoring torch's own pool does not reintroduce the race --
    # the pin is what CHOLMOD reads, and this touches only torch.
    configure_torch_threads()
    clip_loss = build_clip_loss(config) if with_clip else None

    # Seeded after the CLIP weights load, as Venice seeds: building the CLIP
    # modules consumes the global generator, so seeding first would leave the
    # augmentation stream dependent on which models were loaded.
    seed_everything(config.seed)
    model = build_model(clip_loss, config)
    seed_everything(config.seed)

    ds = build_optimizer(model, config, max_iterations).optimize()
    return attach_final_raw_design(ds, model)


def attach_final_raw_design(
    ds: xarray.Dataset,
    model: AdaptivePixelModel,
) -> xarray.Dataset:
    """Record the design parameter as the run left it, unrendered.

    `ds['design']` holds the RENDERED design -- `Environment.render` applies the
    volume constraint, which bounds it to [0, 1]. Two of the reference run's
    artifacts are functions of the raw parameter instead, and neither can be
    recovered from the rendered field:

    * `params['volume_actual']` is Venice's `get_volume_ratio`, the fraction of
      the raw parameter above 0.9 (`models.py` line 837).
    * the final image is the raw parameter resized to `img_width`, THEN clamped
      to [0, 1], then inverted (`models.py` line 815). That order matters, and
      `tests/test_venice_parity._venice_display_field` reproduces it.

    The raw parameter is unbounded -- the reference run's spans -11.78 to 13.12
    -- so clamping and rendering are different operations and the distinction
    matters. Its own dimensions are used because it lives on whichever stage
    grid the run ended on, which is the final grid only when the resolution
    schedule ran to completion.

    Args:
        ds: the run's dataset; modified in place.
        model: the model the run left behind, holding the post-step parameter.

    Returns:
        `ds`, for chaining.
    """
    raw = model.z.detach().cpu().numpy()[0]
    ds['final_design_raw'] = (('raw_y', 'raw_x'), raw)
    density = model.get_physical_density(model.z).detach().cpu().numpy()
    density = np.asarray(density, dtype=np.float32)
    while density.ndim > 2:
        density = density[0]
    if density.shape != raw.shape:
        raise ValueError(
            f'physical density shape {density.shape} does not match raw '
            f'{raw.shape}')
    ds['final_physical_density'] = (('raw_y', 'raw_x'), density)
    return ds


def prompt_slug(prompt: str) -> str:
    """Filesystem token for a CLIP prompt. Empty after stripping is rejected."""
    stripped = prompt.strip()
    if not stripped:
        raise ValueError('prompt must be non-empty')
    slug = re.sub(r'[^a-z0-9]+', '_', stripped.lower()).strip('_')
    if not slug:
        raise ValueError(f'prompt {prompt!r} has no filesystem-safe characters')
    return slug


def motif_scale_run_config(prompt: Optional[str] = None) -> VeniceGoldenConfig:
    """Neutral-init motif-scale config. ``prompt`` overrides CLIP text only."""
    config = venice_250214_motif_scale().to_venice_golden()
    if prompt is None:
        return config
    stripped = prompt.strip()
    if not stripped:
        raise ValueError('prompt must be non-empty')
    if stripped == config.prompt:
        return config
    return dataclasses.replace(config, prompt=stripped)


def motif_scale_results_dir(
    prompt: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Skeletons keeps the existing dir; any other prompt gets its own folder."""
    base = repo_root / 'script' / 'resources' / 'results'
    if prompt.strip() == GOLDEN.prompt:
        return base / 'clip_motif_scale_neutral_init'
    return base / f'clip_motif_scale_neutral_init_{prompt_slug(prompt)}'


def venice_volume_ratio(field, threshold: float = 0.9) -> float:
    """Venice's `get_volume_ratio`: the fraction of `field` above `threshold`.

    A transcription of `models.py` line 837, and deliberately not a mean
    density: it is a strict-inequality COUNT over the raw parameter, divided by
    the element count. On a field that is not saturated the two disagree badly
    -- the reference run logged a volume ratio of 0.3148 for a rendered mean
    density that the volume constraint holds at 0.30 by construction, so
    substituting the mean would turn this pin into a restatement of `volfrac`.

    Args:
        field: the raw design parameter, as an array or tensor.
        threshold: the density a pixel must exceed to count as filled.

    Returns:
        The filled fraction, in [0, 1].
    """
    values = np.asarray(field)
    return float((values > threshold).sum()) / values.size


def save_replay_images(ds: xarray.Dataset, output_dir: Path) -> tuple[Path, Path]:
    """Write the replay design next to Venice's own final image.

    The replay is shown the way Venice saved it: resize the raw unbounded
    parameter's short edge to `img_width`, then clamp to [0, 1], then invert
    so material is black. That is the same round trip the parity harness uses,
    so the PNG is a visual of the tensor the tests actually compare -- not a
    second, prettier rendering of the volume-constrained field.

    Args:
        ds: a dataset that has already gone through `attach_final_raw_design`.
        output_dir: `script/test_results_pytorch/` in a normal replay.

    Returns:
        `(replay_path, comparison_path)`.
    """
    from PIL import Image, ImageDraw

    from neural_structural_optimization.models.loss_clip import _resize_short_side

    raw = np.ascontiguousarray(ds['final_design_raw'].values, dtype=np.float32)
    resized = _resize_short_side(
        torch.as_tensor(raw)[None, None], GOLDEN.clip_resize_short_side,
    ).clamp(0.0, 1.0)
    replay = Image.fromarray(
        (255.0 * (1.0 - resized[0, 0].cpu().numpy())).clip(0, 255).astype(np.uint8),
        mode='L',
    )
    replay_path = output_dir / 'venice_golden_250214_replay.png'
    replay.save(replay_path)

    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    if replay.size != reference.size:
        replay = replay.resize(reference.size, Image.Resampling.NEAREST)

    gap, label_h = 16, 28
    width, height = reference.size
    canvas = Image.new('L', (width * 2 + gap, height + label_h), 255)
    canvas.paste(reference, (0, label_h))
    canvas.paste(replay, (width + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), 'Venice 250214 (reference)', fill=0)
    draw.text((width + gap + 8, 6), 'Hardfork replay', fill=0)
    comparison_path = output_dir / 'venice_golden_250214_comparison.png'
    canvas.save(comparison_path)
    return replay_path, comparison_path


def _venice_display_raw(ds: xarray.Dataset, size: tuple[int, int]):
    """Venice JPEG round trip of the raw design, sized to the reference."""
    from PIL import Image

    from neural_structural_optimization.models.loss_clip import _resize_short_side

    raw = np.ascontiguousarray(ds['final_design_raw'].values, dtype=np.float32)
    resized = _resize_short_side(
        torch.as_tensor(raw)[None, None], GOLDEN.clip_resize_short_side,
    ).clamp(0.0, 1.0)
    image = Image.fromarray(
        (255.0 * (1.0 - resized[0, 0].cpu().numpy())).clip(0, 255).astype(np.uint8),
        mode='L',
    )
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def _ink_black(field: np.ndarray, size: tuple[int, int]):
    """Material=black panel for a [0, 1] field, sized to the reference JPEG."""
    from PIL import Image

    arr = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(
        (255.0 * (1.0 - arr)).clip(0, 255).astype(np.uint8), mode='L')
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def save_motif_scale_look(
    ds: xarray.Dataset,
    output_dir: Path,
    *,
    replay_title: str = 'Motif-scale CLIP, neutral init',
    extra_panels: tuple[tuple[str, Path], ...] = (),
) -> tuple[Path, Path]:
    """Write the no-occupancy motif-scale look next to Venice and any priors."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    size = reference.size
    replay = _venice_display_raw(ds, size)
    replay_path = output_dir / 'sketch_run.png'
    replay.save(replay_path)

    panels: list[tuple[str, Image.Image]] = [
        ('Venice 250214 RRC', reference),
        (replay_title, replay),
    ]
    if 'final_physical_density' in ds:
        density = np.asarray(ds['final_physical_density'].values, dtype=np.float32)
        density_img = _ink_black(density, size)
        density_img.save(output_dir / 'physical_density.png')
        panels.append(('Physical density', density_img))
    for title, path in extra_panels:
        if path.is_file():
            panels.append((
                title,
                Image.open(path).convert('L').resize(size, Image.Resampling.NEAREST),
            ))
    gap, label_h = 16, 28
    width, height = size
    canvas = Image.new(
        'L',
        (width * len(panels) + gap * (len(panels) - 1), height + label_h),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for title, image in panels:
        canvas.paste(image, (x, label_h))
        draw.text((x + 8, 6), title, fill=0)
        x += width + gap
    comparison_path = output_dir / 'comparison.png'
    canvas.save(comparison_path)
    return replay_path, comparison_path


def _last_clip_motif_terms(ds: xarray.Dataset) -> dict:
    return {
        str(name): float(ds[name][-1])
        for name in ds.data_vars
        if str(name).startswith('clip_motif_')
    }


def trajectory(ds: xarray.Dataset) -> dict:
    """Extract the per-step term breakdown as plain lists, for saving."""
    payload = {
        'converged': bool(ds.attrs['converged']),
        'resize_steps': [int(s) for s in ds.attrs['resize_steps']],
        'volume_actual': venice_volume_ratio(ds['final_design_raw'].values),
        'compliance_loss': [float(v) for v in ds['compliance'].values],
        'clip_loss': [float(v) for v in ds['clip_loss'].values],
        'clip_loss_raw': [float(v) for v in ds['clip_loss_raw'].values],
        'clip_weight': [float(v) for v in ds['clip_weight'].values],
        'total_loss': [float(v) for v in ds['loss'].values],
    }
    motif = {
        name: [float(v) for v in ds[name].values]
        for name in ds.data_vars
        if str(name).startswith('clip_motif_')
    }
    if motif:
        payload.update(motif)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    """Replay the golden run and print its final point beside the reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--motif-scale',
        action='store_true',
        help=(
            'Venice CLIP plus physical-scale crops; no sketch occupancy. '
            'Neutral init (no Venice frame image). Default output is '
            'script/resources/results/clip_motif_scale_neutral_init/ for '
            'the skeletons prompt; other prompts get a sibling directory.'),
    )
    parser.add_argument(
        '--prompt',
        help=(
            'CLIP text prompt. Requires --motif-scale. Default is the '
            'preset prompt (skeletons). GOLDEN replay ignores this.'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='result directory (relative paths resolve from the repository root)',
    )
    args = parser.parse_args(argv)
    if args.prompt is not None and not args.motif_scale:
        parser.error('--prompt requires --motif-scale')

    if args.motif_scale:
        config = motif_scale_run_config(args.prompt)
        default_dir = motif_scale_results_dir(config.prompt)
    else:
        config = GOLDEN
        default_dir = REPO_ROOT / 'script' / 'test_results_pytorch'

    output_dir = args.output_dir or default_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'Running Venice 250214 {config.prompt!r}'
        + (' + motif-scale CLIP (no occupancy)' if args.motif_scale else '')
        + '...')
    ds = run(config)

    output_path = output_dir / REPLAY_FILENAME
    output_path.write_text(json.dumps(trajectory(ds), indent=2))
    print(f'\nWrote replay trajectory to {output_path}')
    if args.motif_scale:
        skeletons_prior = (
            REPO_ROOT / 'script' / 'resources' / 'results'
            / 'clip_motif_scale_neutral_init' / 'sketch_run.png')
        extra_panels = ()
        if config.prompt != GOLDEN.prompt and skeletons_prior.is_file():
            extra_panels = (
                ('Neutral-init skeletons (prior)', skeletons_prior),
            )
        replay_image, comparison_image = save_motif_scale_look(
            ds,
            output_dir,
            replay_title=f'Motif-scale CLIP, {config.prompt}',
            extra_panels=extra_panels,
        )
        summary = {
            'clip_prompt': config.prompt,
            'motif_scale': True,
            'occupancy': False,
            'neutral_init': config.neutral_init,
            'init_noise_amp': config.init_noise_amp,
            'union_load_sites': config.union_load_sites,
            'motif_scale_fracs': list(config.motif_scale_fracs),
            'motif_scale_crops': config.motif_scale_crops,
            'motif_scale_weight': config.motif_scale_weight,
            'steps': int(ds.sizes['step']),
            'converged': bool(ds.attrs['converged']),
            'resize_steps': [int(s) for s in ds.attrs['resize_steps']],
            'volume_actual': venice_volume_ratio(ds['final_design_raw'].values),
            'mean_physical_density': (
                float(np.mean(ds['final_physical_density'].values))
                if 'final_physical_density' in ds else None),
            'compliance': float(ds['compliance'][-1]),
            'clip_loss': float(ds['clip_loss'][-1]),
            'clip_loss_raw': float(ds['clip_loss_raw'][-1]),
            'total_loss': float(ds['loss'][-1]),
            **_last_clip_motif_terms(ds),
            'paths': {
                'replay': str(replay_image),
                'comparison': str(comparison_image),
                'physical_density': str(output_dir / 'physical_density.png'),
            },
        }
        (output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
    else:
        replay_image, comparison_image = save_replay_images(ds, output_dir)
    print(f'Wrote replay design to {replay_image}')
    print(f'Wrote side-by-side comparison to {comparison_image}')

    steps = int(ds.sizes['step'])
    print(f'\nStopped after {steps} steps '
          f"(converged: {bool(ds.attrs['converged'])}, "
          f"upsamples at {ds.attrs['resize_steps']}).")
    print(f'Reference stopped at {GOLDEN_FINAL_ITERATION}.\n')

    replay = {
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': float(ds['clip_loss'][-1]),
        'clip_loss_raw': float(ds['clip_loss_raw'][-1]),
        'clip_weight': float(ds['clip_weight'][-1]),
        'total': float(ds['loss'][-1]),
        'volume_actual': venice_volume_ratio(ds['final_design_raw'].values),
    }
    print(f'{"term":<16}{"replay":>18}{"reference":>18}{"rel. diff":>14}')
    for name, reference in GOLDEN_FINAL.items():
        value = replay[name]
        print(f'{name:<16}{value:>18.6f}{reference:>18.6f}'
              f'{abs(value - reference) / abs(reference):>14.2%}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
