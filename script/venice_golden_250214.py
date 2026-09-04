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

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import configure_torch_threads
from neural_structural_optimization.models import AdaptivePixelModel, CLIPLoss
from neural_structural_optimization.models.loss_clip import VeniceClipPreset
from neural_structural_optimization.models.model_base import VeniceLossAlgebra
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.train import AdaptiveAdam_Optimizer
from neural_structural_optimization.train.utils import init_weight_with_image

# Copied out of the Venice tree (commit 97f9336) rather than referenced across
# repositories. A golden run that reads its initial condition from a sibling
# checkout is not reproducible: it silently changes meaning when that checkout
# moves, and fails outright when it is absent. The relative path under
# `resources/` is preserved so the provenance stays obvious, and the file is
# byte-identical (sha256 750bb809f38c0f5ef6f9af51f228a860a40287b56efde07e27dc061eff157d98).
GOLDEN_IMAGE_PATH = (
    REPO_ROOT / 'script' / 'resources' / 'input_images' / 'dSketches'
    / 'thick_outer_lins.png')


@dataclass(frozen=True)
class VeniceGoldenConfig:
    """Every knob of the reference run, with the log's values as defaults.

    Attributes are grouped by the object that consumes them: the structural
    problem, the model's resolution schedule, the CLIP loss, the loss algebra
    and the optimizer.
    """

    # Structural problem, at its FINAL resolution. `AdaptivePixelModel` divides
    # width, height and interval down to the schedule's coarse start itself.
    problem_name: str = 'multistory_building'
    width: int = 128
    height: int = 256
    density: float = 0.3
    interval: int = 64
    filter_width: float = 2.0
    # Fixed at 3.0 by `structural.api.specified_task`, which takes no override;
    # kept here so the run asserts the value it was configured for.
    penal: float = 3.0

    # Resolution schedule. Two doublings, so training starts at 32x64.
    resize_num: int = 2
    resize_scale: int = 2

    # CLIP guidance.
    clip_model_name: str = 'ViT-B/32'
    # Venice loads a second, ResNet CLIP for its geometric loss. That loss is
    # off in this run (`clip_rn_alpha` is 0 and `geometric_loss` is absent from
    # `loss_types`, and the log records it as null throughout), and the Venice
    # CLIP path never touches the ResNet trunk, so the variant is inert. RN50
    # stands in for Venice's RN101 purely because it is smaller to load.
    clip_rn_model_name: str = 'RN50'
    prompt: str = 'skeletons'
    num_augs: int = 32
    # Venice resizes the design's shorter edge to `params['img_width']` before
    # cropping; the reference run's 512 gives the CLIP stack a 512x1024 view.
    clip_resize_short_side: int = 512

    # Loss algebra: clip_weight = compliance * clip_alpha, undetached, with the
    # unweighted CLIP term added on top. See `models.model_base`.
    clip_alpha: float = 10.0
    compliance_weight: float = 1.0

    # Optimizer. Adam at a constant rate -- there is no schedule in Venice.
    lr: float = 0.2
    max_iterations: int = 200
    resize_threshold: float = 0.5
    max_resize_iteration: int = 50
    convergence_threshold: float = 0.05

    seed: int = 12
    device: str = 'cpu'
    invert_image: bool = True


GOLDEN = VeniceGoldenConfig()

# The same wiring, sized to run in seconds rather than a quarter of an hour: a
# quarter-scale grid, a quarter of the CLIP view and an eighth of the crops.
# The numbers it produces mean nothing next to the reference log -- it exists to
# prove the preset, the image seeding, the resolution schedule and the term
# breakdown still hold together, which is what a default test run can afford.
SMOKE = VeniceGoldenConfig(
    width=32,
    height=64,
    interval=16,
    num_augs=4,
    clip_resize_short_side=128,
    max_iterations=4,
)

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
    )


def build_model(
    clip_loss: Optional[CLIPLoss] = None,
    config: VeniceGoldenConfig = GOLDEN,
    image_path: Optional[Path] = None,
) -> AdaptivePixelModel:
    """Build the adaptive model, coarse-started and seeded from the image.

    Args:
        clip_loss: the semantic loss, or None for a structure-only replay.
        config: the run configuration.
        image_path: override for the initial image; defaults to the copied
            Venice asset.

    Returns:
        A model at the schedule's coarse resolution whose design parameter is
        the initial image, with the legacy loss algebra enabled.
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
    replay can be compared against the reference log field by field.

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

    return build_optimizer(model, config, max_iterations).optimize()


def trajectory(ds: xarray.Dataset) -> dict:
    """Extract the per-step term breakdown as plain lists, for saving."""
    return {
        'converged': bool(ds.attrs['converged']),
        'resize_steps': [int(s) for s in ds.attrs['resize_steps']],
        'compliance_loss': [float(v) for v in ds['compliance'].values],
        'clip_loss': [float(v) for v in ds['clip_loss'].values],
        'clip_loss_raw': [float(v) for v in ds['clip_loss_raw'].values],
        'clip_weight': [float(v) for v in ds['clip_weight'].values],
        'total_loss': [float(v) for v in ds['loss'].values],
    }


def main() -> int:
    """Replay the golden run and print its final point beside the reference."""
    ds = run()

    output_path = REPO_ROOT / 'script' / 'test_results_pytorch' / REPLAY_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectory(ds), indent=2))
    print(f'\nWrote replay trajectory to {output_path}')

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
    }
    print(f'{"term":<16}{"replay":>18}{"reference":>18}{"rel. diff":>14}')
    for name, reference in GOLDEN_FINAL.items():
        value = replay[name]
        print(f'{name:<16}{value:>18.6f}{reference:>18.6f}'
              f'{abs(value - reference) / abs(reference):>14.2%}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
