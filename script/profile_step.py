"""Profile one Venice CLIP step into physics / CLIP / backward / optimizer.

The Stage 2 leftover gate: a written per-step breakdown on the machine that
runs looks. CLIP-vs-physics split decides whether to pursue an analytic
compliance VJP or to treat ``num_augs`` / encoder / device as the speed knobs
before adding more crop scales.

Default is the smoke preset (coarse grid, 4 augs). ``--full`` times a 128x256
Venice CLIP step with 32 augs after upsampling, which is the Stage 6 look cost.

    PYTHONPATH="$PWD" python script/profile_step.py
    PYTHONPATH="$PWD" python script/profile_step.py --full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_structural_optimization import configure_torch_threads
from neural_structural_optimization.experiment import GOLDEN, SMOKE
from neural_structural_optimization.models.model_base import venice_compat_total_loss


def _load_golden_script():
    import importlib.util
    script = REPO_ROOT / 'script' / 'venice_golden_250214.py'
    spec = importlib.util.spec_from_file_location('venice_golden_250214', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean_std(samples: list[float]) -> dict:
    if not samples:
        return {'mean_s': None, 'std_s': None, 'n': 0}
    if len(samples) == 1:
        return {'mean_s': samples[0], 'std_s': 0.0, 'n': 1}
    return {
        'mean_s': statistics.fmean(samples),
        'std_s': statistics.pstdev(samples),
        'n': len(samples),
    }


def _time_call(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def profile(config, *, upsample_to_full: bool, warmup: int, repeats: int) -> dict:
    configure_torch_threads()
    golden = _load_golden_script()
    clip_loss = golden.build_clip_loss(config)
    model = golden.build_model(clip_loss, config)
    if upsample_to_full:
        while model.can_upsample:
            model.upsample()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    _, height, width = model.shape

    def one_combined_step():
        optimizer.zero_grad(set_to_none=True)
        logits = model()
        terms = model.get_venice_compat_losses(logits)
        terms.total_loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_combined_step()

    combined = _time_call(one_combined_step, repeats)

    physics_fwd, clip_fwd, backward, opt_step = [], [], [], []
    physics_bwd, clip_bwd = [], []
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        logits = model()

        t0 = time.perf_counter()
        structural = model.get_structural_loss(logits)
        physics_fwd.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        semantic = model.get_semantic_loss(logits)
        clip_fwd.append(time.perf_counter() - t1)

        terms = venice_compat_total_loss(
            structural, semantic,
            clip_alpha=config.clip_alpha,
            compliance_weight=config.compliance_weight,
        )
        t2 = time.perf_counter()
        terms.total_loss.backward()
        backward.append(time.perf_counter() - t2)

        t3 = time.perf_counter()
        optimizer.step()
        opt_step.append(time.perf_counter() - t3)

        optimizer.zero_grad(set_to_none=True)
        logits = model()
        structural = model.get_structural_loss(logits)
        t4 = time.perf_counter()
        structural.backward()
        physics_bwd.append(time.perf_counter() - t4)

        optimizer.zero_grad(set_to_none=True)
        logits = model()
        semantic = model.get_semantic_loss(logits)
        t5 = time.perf_counter()
        semantic.backward()
        clip_bwd.append(time.perf_counter() - t5)
        optimizer.zero_grad(set_to_none=True)

    combined_mean = statistics.fmean(combined)
    parts = {
        'physics_forward': _mean_std(physics_fwd),
        'clip_forward': _mean_std(clip_fwd),
        'backward_combined': _mean_std(backward),
        'optimizer_step': _mean_std(opt_step),
        'physics_backward_only': _mean_std(physics_bwd),
        'clip_backward_only': _mean_std(clip_bwd),
        'combined_step': _mean_std(combined),
    }
    clip_share = (
        parts['clip_forward']['mean_s'] + parts['clip_backward_only']['mean_s'])
    physics_share = (
        parts['physics_forward']['mean_s'] + parts['physics_backward_only']['mean_s'])
    if clip_share >= physics_share:
        dominant = 'clip'
        recommendation = (
            'CLIP owns the step. Treat num_augs, encoder, and device as '
            'config before adding more crop scales. Do not start an analytic '
            'compliance VJP from this profile.')
    else:
        dominant = 'physics'
        recommendation = (
            'Physics owns the step. Analytic compliance VJP is the next '
            'speed play, gated by a finite-difference check against HIPS.')
    return {
        'preset': 'venice_250214' if upsample_to_full else 'venice_250214_smoke',
        'grid': f'{width}x{height}',
        'num_augs': config.num_augs,
        'device': config.device,
        'clip_model_name': config.clip_model_name,
        'warmup': warmup,
        'repeats': repeats,
        'parts_s': parts,
        'clip_share_s': clip_share,
        'physics_share_s': physics_share,
        'combined_step_mean_s': combined_mean,
        'dominant': dominant,
        'recommendation': recommendation,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--full', action='store_true',
        help='profile a 128x256 Venice CLIP step (32 augs), not smoke')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument(
        '--output', type=Path, default=None,
        help='JSON path (default: script/resources/results/stage2_profile_step.json)')
    args = parser.parse_args(argv)

    config = GOLDEN if args.full else SMOKE
    repeats = 1 if args.full else args.repeats
    payload = profile(
        config, upsample_to_full=args.full, warmup=args.warmup, repeats=repeats)

    output = args.output or (
        REPO_ROOT / 'script' / 'resources' / 'results' / 'stage2_profile_step.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f'\nWrote {output}')
    print(f"dominant={payload['dominant']}")
    print(payload['recommendation'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
