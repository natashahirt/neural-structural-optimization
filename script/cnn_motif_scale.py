"""Motif-scale CLIP on the CNN reparameterization instead of raw pixels.

The pixel runs keep reaching the same conclusion: CLIP writes ORNAMENT, not
STRUCTURE. Neutral init, a longer coarse stage, and two different prompts all
landed on the same three-bay frame with a motif painted over it. The suspicion
this script tests is that the failure is the BASIS rather than the signal.

Under `AdaptivePixelModel` every element of the design is an independent Adam
parameter, so a speckle costs one update and a coherent large member costs
thousands of coordinated ones. CLIP's gradient -- averaged over 32 random
crops with flips and rotations -- is diffuse and has no low-frequency
agreement, so the cheap move is the one it takes. This is the same failure
raw-pixel CLIP optimization had in 2021, and the fix that worked then was not
a better prompt or a bigger weight: it was decoding the image from a latent
through a frozen generator (VQGAN+CLIP), so that the only moves available were
coherent ones. `CNNModel` is this repository's analogue -- a latent vector
through a dense layer to a small base grid, then conv/upsample stages to the
design field -- with an architectural prior in place of a learned one.

What is held fixed against `clip_motif_scale_neutral_init_*`:

* the problem, resolution, volume fraction and filter (`golden_structural_params`),
* the CLIP loss in full: prompt, Venice preset, augmentation count, and the
  physical motif-scale crops (`build_clip_loss`),
* the Venice loss algebra, so `clip_weight = compliance * clip_alpha` exactly
  as before.

What necessarily changes, and why:

* **The resolution ladder is gone.** Venice's 32x64 -> 64x128 -> 128x256
  schedule is a hand-built substitute for a multi-scale prior, which is
  precisely what the CNN supplies intrinsically. Rebuilding it on top would
  confound the variable under test, so the CNN runs at the final grid
  throughout, and `AdaptiveAdam_Optimizer` -- which hard-rejects any model that
  is not `AdaptivePixelModel` -- gives way to plain `Adam_Optimizer`.
* **The learning rate is recalibrated.** Venice's 0.2 is tuned for a
  density-like field in pixel space; conv weights are not that. Calibrate
  structure-only first (`--no-clip`) and only spend a CLIP run on a learning
  rate that reaches compliance near the pixel baseline, or the run reports the
  learning rate rather than the hypothesis.
* **Neutral init is implicit.** `init_weight_neutral` seeds `model.z`, which on
  a `CNNModel` is the 128-dim LATENT and not a design field. The property that
  mattered -- no pre-authored architectural frame -- holds anyway, because the
  latent is seeded from noise and the volume constraint pins mean density.

Run it with::

    PYTHONPATH="$PWD" python script/cnn_motif_scale.py --no-clip --lr 1e-2
    PYTHONPATH="$PWD" python script/cnn_motif_scale.py --prompt "butterfly wing venation"
"""

import argparse
import importlib.util
import json
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
from neural_structural_optimization.experiment import VeniceGoldenConfig
from neural_structural_optimization.models import CNNModel
from neural_structural_optimization.models.config import (
    DEFAULT_CONV_UPSAMPLE,
    DEFAULT_LATENT_SIZE,
)
from neural_structural_optimization.models.model_base import VeniceLossAlgebra
from neural_structural_optimization.train import Adam_Optimizer

# Gradient steps, chosen to match the pixel venation run's 128 so the two are
# read side by side. `Adam_Optimizer` iterates `max_iterations + 1` times, so
# the count it is handed is one less than the steps it takes.
DEFAULT_STEPS = 128

# `Adam_Optimizer` cosine-decays from `lr_init` to `lr_final` after a warmup.
# These are its own defaults, restated because they are the calibration's
# starting point rather than an incidental inheritance.
DEFAULT_LR_INIT = 1e-2
DEFAULT_LR_FINAL = 3e-3


def _load_golden():
    """Import the golden script from ``script/``, which is not a package.

    The same loader `experiment._load_golden_script` uses. The golden run's
    builders are reused rather than restated so that the CLIP loss, structural
    problem and result artifacts here cannot drift away from the pixel runs
    this is meant to be compared against.
    """
    script = REPO_ROOT / 'script' / 'venice_golden_250214.py'
    spec = importlib.util.spec_from_file_location('venice_golden_250214', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cnn_base_grid(
    config: VeniceGoldenConfig,
    conv_upsample=DEFAULT_CONV_UPSAMPLE,
) -> tuple[int, int]:
    """The decoder's base grid, `(rows, cols)` before any conv upsampling.

    Reported because it, together with the latent size, IS the hypothesis: the
    design field is reachable only as a smooth expansion of this many cells, so
    per-element speckle is not in the model's range at any learning rate.

    Args:
        config: the run configuration, for the final grid.
        conv_upsample: per-stage upsample factors; their product divides the
            grid.

    Returns:
        `(height // product, width // product)`.

    Raises:
        ValueError: if the final grid is not divisible by the product, which is
            the same condition `CNNModel._compute_base_sizes` rejects.
    """
    total_up = 1
    for factor in conv_upsample:
        total_up *= int(factor)
    if config.height % total_up or config.width % total_up:
        raise ValueError(
            f'grid {config.height}x{config.width} is not divisible by '
            f'product(conv_upsample)={total_up}')
    return config.height // total_up, config.width // total_up


def build_cnn_model(
    clip_loss=None,
    config: Optional[VeniceGoldenConfig] = None,
    *,
    latent_size: int = DEFAULT_LATENT_SIZE,
    golden=None,
) -> CNNModel:
    """Build the CNN decoder on the golden structural problem.

    Args:
        clip_loss: the semantic loss, or None for a structure-only calibration.
        config: the run configuration; the motif-scale preset by default.
        latent_size: width of the latent vector, the decoder's bottleneck.
        golden: the loaded golden module, to avoid re-importing it.

    Returns:
        A `CNNModel` at the FINAL grid -- there is no coarse start -- with the
        Venice loss algebra enabled.

    Raises:
        ValueError: if the physics penalization does not match the config, the
            same guard the golden builder applies.
    """
    golden = golden or _load_golden()
    config = config if config is not None else golden.motif_scale_run_config()
    model = CNNModel(
        structural_params=golden.golden_structural_params(config),
        clip_loss=clip_loss,
        latent_size=latent_size,
        seed=config.seed,
    )
    if model.args['penal'] != config.penal:
        raise ValueError(
            f"physics penal is {model.args['penal']}, not the configured "
            f'{config.penal}; this is not the golden problem.')

    # Same seam the golden builder flips, for the same reason: the model
    # subclasses do not forward `venice_compat` through their constructors.
    # Without it `clip_weight` would follow Adam's inverse-normalized coupling
    # instead of Venice's `compliance * clip_alpha`, and the run would not be
    # comparable to the pixel one.
    model.enable_venice_compat_loss(VeniceLossAlgebra(
        clip_alpha=config.clip_alpha,
        compliance_weight=config.compliance_weight,
    ))
    return model


def build_cnn_optimizer(
    model: CNNModel,
    config: VeniceGoldenConfig,
    *,
    steps: int = DEFAULT_STEPS,
    lr_init: float = DEFAULT_LR_INIT,
    lr_final: float = DEFAULT_LR_FINAL,
) -> Adam_Optimizer:
    """Build the optimizer for the CNN path.

    `clip_alpha` and `clip_weight_max` are deliberately NOT passed: the model's
    algebra owns the CLIP coupling, and `Adam_Optimizer` refuses both rather
    than silently applying one formula while the caller expects the other.

    Args:
        model: the CNN whose parameters are optimized.
        config: the run configuration, for `compliance_weight`.
        steps: total gradient steps.
        lr_init: peak learning rate after warmup.
        lr_final: floor of the cosine decay.

    Returns:
        A configured `Adam_Optimizer`.
    """
    return Adam_Optimizer(
        model,
        max_iterations=steps - 1,
        lr_init=lr_init,
        lr_final=lr_final,
        compliance_weight=config.compliance_weight,
    )


def attach_final_cnn_design(
    ds: xarray.Dataset,
    model: CNNModel,
) -> xarray.Dataset:
    """Record the final design field, raw and physical.

    The pixel runs read `model.z`, which on a `CNNModel` is the LATENT VECTOR
    rather than a design field -- `golden.attach_final_raw_design` would happily
    store a 128-element strip and every downstream image would be nonsense. The
    decoder output is the corresponding object here, so it is recomputed.

    Args:
        ds: the run's dataset; modified in place.
        model: the model the run left behind.

    Returns:
        `ds`, for chaining.
    """
    with torch.no_grad():
        logits = model.forward()
        density = model.get_physical_density(logits)
    raw = np.asarray(logits.detach().cpu().numpy(), dtype=np.float32)
    density = np.asarray(density.detach().cpu().numpy(), dtype=np.float32)
    while raw.ndim > 2:
        raw = raw[0]
    while density.ndim > 2:
        density = density[0]
    if density.shape != raw.shape:
        raise ValueError(
            f'physical density shape {density.shape} does not match raw '
            f'{raw.shape}')
    ds['final_design_raw'] = (('raw_y', 'raw_x'), raw)
    ds['final_physical_density'] = (('raw_y', 'raw_x'), density)
    ds.attrs.update(_final_motif_scale_terms(model))

    # `golden.trajectory` reads both, and `Adam_Optimizer` sets neither: there
    # is no resolution ladder on this path and no compliance-based stopping
    # rule, so the run always takes its full step budget.
    ds.attrs.setdefault('resize_steps', [])
    ds.attrs.setdefault('converged', 0)
    return ds


def run(
    config: Optional[VeniceGoldenConfig] = None,
    *,
    steps: int = DEFAULT_STEPS,
    lr_init: float = DEFAULT_LR_INIT,
    lr_final: float = DEFAULT_LR_FINAL,
    latent_size: int = DEFAULT_LATENT_SIZE,
    with_clip: bool = True,
    golden=None,
) -> xarray.Dataset:
    """Optimize the CNN decoder and return the run's dataset.

    Args:
        config: the run configuration; the motif-scale preset by default.
        steps: total gradient steps.
        lr_init: peak learning rate after warmup.
        lr_final: floor of the cosine decay.
        latent_size: width of the latent vector.
        with_clip: build the CLIP loss. False gives the structure-only
            calibration, which is also the no-CLIP control for this basis.
        golden: the loaded golden module, to avoid re-importing it.

    Returns:
        The dataset, carrying the per-step term breakdown plus the final raw
        and physical design fields.
    """
    # The package pins OMP_NUM_THREADS=1 at import so CHOLMOD cannot race,
    # which leaves torch single-threaded until this restores its pool.
    configure_torch_threads()
    golden = golden or _load_golden()
    config = config if config is not None else golden.motif_scale_run_config()

    clip_loss = golden.build_clip_loss(config) if with_clip else None

    # Seeded after the CLIP weights load, as the golden run seeds: building the
    # CLIP modules draws from the global generator, so seeding first would make
    # the augmentation stream depend on which models were loaded.
    golden.seed_everything(config.seed)
    model = build_cnn_model(
        clip_loss, config, latent_size=latent_size, golden=golden)
    golden.seed_everything(config.seed)

    ds = build_cnn_optimizer(
        model, config, steps=steps, lr_init=lr_init, lr_final=lr_final,
    ).optimize()
    return attach_final_cnn_design(ds, model)


def cnn_results_dir(
    prompt: str,
    with_clip: bool,
    repo_root: Path = REPO_ROOT,
    golden=None,
) -> Path:
    """Result directory for a CNN run, a sibling of the pixel motif dirs."""
    golden = golden or _load_golden()
    base = repo_root / 'script' / 'resources' / 'results'
    if not with_clip:
        return base / 'cnn_structure_only'
    return base / f'cnn_motif_scale_{golden.prompt_slug(prompt)}'


def _final_motif_scale_terms(model: CNNModel) -> dict:
    """Per-scale CLIP contributions recorded on the model's last forward.

    `Adam_Optimizer` keeps no per-step snapshots of these -- only
    `AdaptiveAdam_Optimizer` does -- so the final step's values are what is
    available, which is what the summary reports.
    """
    losses = getattr(getattr(model, 'clip_loss', None),
                     'last_motif_scale_losses', None)
    if not losses:
        return {}
    return {f'clip_motif_{key}': float(value) for key, value in losses.items()}


def main(argv: Optional[list[str]] = None) -> int:
    """Run the CNN motif-scale experiment and write its artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            'Motif-scale CLIP on the CNN reparameterization. Holds the '
            'problem and the CLIP loss fixed against the pixel runs and '
            'changes only the parameterization.'))
    parser.add_argument(
        '--prompt',
        help='CLIP text prompt. Defaults to the motif-scale preset prompt.')
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help=(
            'structure-only run: learning-rate calibration, and the no-CLIP '
            'control for this basis. Compare compliance against the pixel '
            'baseline before trusting any CLIP run at this learning rate.'))
    parser.add_argument(
        '--steps', type=int, default=DEFAULT_STEPS,
        help=f'gradient steps (default {DEFAULT_STEPS}, the pixel run\'s count)')
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_LR_INIT,
        help=f'peak learning rate after warmup (default {DEFAULT_LR_INIT})')
    parser.add_argument(
        '--lr-final', type=float, default=None,
        help='cosine-decay floor (default: 0.3 * --lr)')
    parser.add_argument(
        '--latent-size', type=int, default=DEFAULT_LATENT_SIZE,
        help=f'decoder bottleneck width (default {DEFAULT_LATENT_SIZE})')
    parser.add_argument(
        '--output-dir', type=Path,
        help='result directory (relative paths resolve from the repository root)')
    args = parser.parse_args(argv)

    golden = _load_golden()
    with_clip = not args.no_clip
    config = golden.motif_scale_run_config(args.prompt)
    lr_final = args.lr_final if args.lr_final is not None else 0.3 * args.lr

    output_dir = args.output_dir or cnn_results_dir(
        config.prompt, with_clip, golden=golden)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows, base_cols = cnn_base_grid(config)
    print(
        f'CNN reparameterization, {config.height}x{config.width} grid from a '
        f'{base_rows}x{base_cols} base and a {args.latent_size}-dim latent.')
    print(
        f'  prompt: {config.prompt!r}' if with_clip
        else '  structure only (no CLIP)')
    print(f'  {args.steps} steps, lr {args.lr:g} -> {lr_final:g}')

    configure_torch_threads()
    golden.seed_everything(config.seed)
    clip_loss = golden.build_clip_loss(config) if with_clip else None
    golden.seed_everything(config.seed)
    model = build_cnn_model(
        clip_loss, config, latent_size=args.latent_size, golden=golden)
    golden.seed_everything(config.seed)
    ds = attach_final_cnn_design(
        build_cnn_optimizer(
            model, config, steps=args.steps, lr_init=args.lr,
            lr_final=lr_final,
        ).optimize(),
        model,
    )

    (output_dir / golden.REPLAY_FILENAME).write_text(
        json.dumps(golden.trajectory(ds), indent=2))

    # The pixel venation run, so the two parameterizations sit in one strip.
    prior = (
        REPO_ROOT / 'script' / 'resources' / 'results'
        / f'clip_motif_scale_neutral_init_{golden.prompt_slug(config.prompt)}'
        / 'sketch_run.png')
    extra_panels = (
        (('Pixel basis (prior)', prior),) if with_clip and prior.is_file()
        else ())
    replay_image, comparison_image = golden.save_motif_scale_look(
        ds,
        output_dir,
        replay_title=(
            f'CNN basis, {config.prompt}' if with_clip else 'CNN basis, no CLIP'),
        extra_panels=extra_panels,
    )

    summary = {
        'parameterization': 'cnn',
        'latent_size': args.latent_size,
        'base_grid': [base_rows, base_cols],
        'clip_prompt': config.prompt if with_clip else None,
        'motif_scale': with_clip,
        'motif_scale_fracs': list(config.motif_scale_fracs),
        'motif_scale_weight': config.motif_scale_weight,
        'steps': int(ds.sizes['step']),
        'lr_init': args.lr,
        'lr_final': lr_final,
        'volume_actual': golden.venice_volume_ratio(
            ds['final_design_raw'].values),
        'mean_physical_density': float(
            np.mean(ds['final_physical_density'].values)),
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': float(ds['clip_loss'][-1]),
        'clip_loss_raw': float(ds['clip_loss_raw'][-1]),
        'total_loss': float(ds['loss'][-1]),
        'paths': {
            'replay': str(replay_image),
            'comparison': str(comparison_image),
            'physical_density': str(output_dir / 'physical_density.png'),
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'\nWrote comparison to {comparison_image}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
