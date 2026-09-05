"""Two-pass motif-layout self-distillation.

Pass 1: generate or reuse a raw motif-scale CLIP teacher (the compelling
no-occupancy look). Persist its raw design, physical density, and metrics.

Pass 2: freeze a soft multiscale occupancy scaffold from teacher ink, seed the
student from the teacher, keep raw multi-scale CLIP decoration, and anneal the
spatial mass prior from strong early layout guidance to weaker late guidance.

    PYTHONPATH="$PWD" python script/motif_layout_distillation.py

Does not overwrite ``clip_motif_scale_no_occupancy`` or
``clip_motif_scale_physical_density_no_occupancy``. Empty layout config remains
bit-identical to Venice; this script is the opt-in student path.
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

from neural_structural_optimization.experiment import (
    GOLDEN,
    venice_250214_motif_layout,
    venice_250214_motif_scale,
)
from neural_structural_optimization.models.loss_sketch import (
    apply_motif_layout_config,
    load_site_mask,
    mass_fraction_on_occupancy,
    motif_layout_scaffold,
    sketch_mass_prior_loss,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / 'clip_motif_layout_no_occupancy')
PRIOR_TEACHER_DIR = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / 'clip_motif_scale_no_occupancy')
GOLDEN_FINAL_IMAGE_PATH = (
    REPO_ROOT / 'script' / 'resources' / 'results'
    / '250214_skeleton_loss_test_balanced_dynamic' / '2_final'
    / ('0_final-P_multistory_building-M_Ada-T_skeletons-W_128-H_256-V_0.30'
       '-LR_0.20-CW_10-ID_01-CompL_74.00-ClipL_0.38-VA_0.31'
       '-balanced_dynamic.jpg'))
TEACHER_RAW_NAME = 'final_design_raw.npy'
TEACHER_DENSITY_NAME = 'physical_density.npy'

# Recorded teacher baseline (expe-2293aefa, clip_motif_scale_no_occupancy).
TEACHER_BASELINE_COMPLIANCE = 80.04
TEACHER_BASELINE_MEAN_DENSITY = 0.30


def _load_golden_script():
    script = REPO_ROOT / 'script' / 'venice_golden_250214.py'
    spec = importlib.util.spec_from_file_location('venice_golden_250214', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ink_black(field: np.ndarray, size: tuple[int, int]) -> Image.Image:
    arr = np.clip(np.asarray(field, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(
        (255.0 * (1.0 - arr)).clip(0, 255).astype(np.uint8), mode='L')
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def _venice_display(raw: np.ndarray, size: tuple[int, int]) -> Image.Image:
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


def _squeeze2d(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field)
    if arr.ndim == 3:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(f'expected a 2-D field, got {arr.shape}')
    return arr


def _last_clip_motif_terms(ds) -> dict:
    out = {}
    for name in ds.data_vars:
        key = str(name)
        if key.startswith('clip_motif_'):
            out[key] = float(ds[name][-1])
    return out


def _metrics_from_dataset(ds, golden_mod, raw: np.ndarray) -> dict:
    payload = {
        'steps': int(ds.sizes['step']),
        'converged': bool(ds.attrs['converged']),
        'resize_steps': [int(s) for s in ds.attrs['resize_steps']],
        'volume_actual': golden_mod.venice_volume_ratio(raw),
        'compliance': float(ds['compliance'][-1]),
        'clip_loss': float(ds['clip_loss'][-1]),
        'clip_loss_raw': float(ds['clip_loss_raw'][-1]),
        'total_loss': float(ds['loss'][-1]),
        **_last_clip_motif_terms(ds),
    }
    return payload


def _physical_density(model) -> np.ndarray:
    density = model.get_physical_density(model()).detach().cpu().numpy()
    return _squeeze2d(density)


def _teacher_paths(teacher_dir: Path) -> tuple[Path, Path]:
    return teacher_dir / TEACHER_RAW_NAME, teacher_dir / TEACHER_DENSITY_NAME


def load_teacher_fields(teacher_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    raw_path, density_path = _teacher_paths(teacher_dir)
    if not raw_path.is_file():
        return None
    raw = _squeeze2d(np.load(raw_path))
    if density_path.is_file():
        density = _squeeze2d(np.load(density_path))
        return raw, density
    return raw, None


def persist_teacher_fields(
    teacher_dir: Path, raw: np.ndarray, density: np.ndarray,
) -> None:
    teacher_dir.mkdir(parents=True, exist_ok=True)
    np.save(teacher_dir / TEACHER_RAW_NAME, np.ascontiguousarray(raw))
    np.save(teacher_dir / TEACHER_DENSITY_NAME, np.ascontiguousarray(density))


def run_teacher(
    *,
    golden_mod,
    max_iterations: int | None = None,
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    config = venice_250214_motif_scale().to_venice_golden()
    from neural_structural_optimization import configure_torch_threads
    configure_torch_threads()
    clip_loss = golden_mod.build_clip_loss(config)
    golden_mod.seed_everything(config.seed)
    model = golden_mod.build_model(clip_loss, config)
    golden_mod.seed_everything(config.seed)
    ds = golden_mod.attach_final_raw_design(
        golden_mod.build_optimizer(
            model, config, max_iterations=max_iterations).optimize(),
        model)
    raw = np.ascontiguousarray(ds['final_design_raw'].values, dtype=np.float32)
    density = _physical_density(model)
    metrics = _metrics_from_dataset(ds, golden_mod, raw)
    metrics['mean_physical_density'] = float(np.mean(density))
    metrics['motif_scale_fracs'] = list(config.motif_scale_fracs)
    return ds, raw, density, metrics


def extract_scaffold(raw: np.ndarray, layout, scale_fracs: tuple[float, ...]) -> np.ndarray:
    return motif_layout_scaffold(
        raw,
        scale_fracs=scale_fracs,
        threshold=float(layout.threshold),
        envelope_sigma_frac=float(layout.envelope_sigma_frac),
        combine=str(layout.combine),
    )


def run_student(
    *,
    golden_mod,
    teacher_raw: np.ndarray,
    max_iterations: int | None = None,
):
    experiment = venice_250214_motif_layout()
    config = experiment.to_venice_golden()
    scale_fracs = experiment.resolved_layout_scale_fracs()
    from neural_structural_optimization import configure_torch_threads
    configure_torch_threads()
    clip_loss = golden_mod.build_clip_loss(config)
    golden_mod.seed_everything(config.seed)
    model = golden_mod.build_model(clip_loss, config)
    apply_motif_layout_config(
        model,
        experiment.layout,
        teacher_raw,
        scale_fracs=scale_fracs,
    )
    golden_mod.seed_everything(config.seed)
    ds = golden_mod.attach_final_raw_design(
        golden_mod.build_optimizer(
            model, config, max_iterations=max_iterations).optimize(),
        model)
    return experiment, model, ds


def save_layout_look(
    *,
    output_dir: Path,
    teacher_raw: np.ndarray,
    teacher_density: np.ndarray,
    scaffold: np.ndarray,
    student_raw: np.ndarray,
    student_density: np.ndarray,
) -> dict[str, Path]:
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    size = reference.size
    teacher_raw_img = _venice_display(teacher_raw, size)
    teacher_density_img = _ink_black(teacher_density, size)
    scaffold_img = _ink_black(scaffold, size)
    student_raw_img = _venice_display(student_raw, size)
    student_density_img = _ink_black(student_density, size)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'teacher_raw': output_dir / 'teacher_raw.png',
        'teacher_density': output_dir / 'teacher_physical_density.png',
        'scaffold': output_dir / 'scaffold.png',
        'student_raw': output_dir / 'sketch_run.png',
        'student_density': output_dir / 'physical_density.png',
        'comparison': output_dir / 'comparison.png',
    }
    teacher_raw_img.save(paths['teacher_raw'])
    teacher_density_img.save(paths['teacher_density'])
    scaffold_img.save(paths['scaffold'])
    student_raw_img.save(paths['student_raw'])
    student_density_img.save(paths['student_density'])
    _hstack(
        [
            ('Venice 250214 RRC', reference),
            ('Teacher raw', teacher_raw_img),
            ('Teacher physical density', teacher_density_img),
            ('Soft scaffold', scaffold_img),
            ('Student raw', student_raw_img),
            ('Student physical density', student_density_img),
        ],
        paths['comparison'],
    )
    return paths


def run_distillation(
    *,
    output_dir: Path,
    teacher_dir: Path | None = None,
    max_iterations: int | None = None,
    scaffold_only: bool = False,
) -> dict:
    import torch

    golden_mod = _load_golden_script()
    experiment = venice_250214_motif_layout()
    scale_fracs = experiment.resolved_layout_scale_fracs()
    layout = experiment.layout

    persist_dir = output_dir / 'teacher'
    reuse_dir = teacher_dir if teacher_dir is not None else persist_dir
    loaded = load_teacher_fields(reuse_dir)
    teacher_reused = loaded is not None and loaded[1] is not None
    if teacher_reused:
        teacher_raw, teacher_density = loaded
        teacher_metrics = {
            'reused': True,
            'teacher_dir': str(reuse_dir),
            'mean_physical_density': float(np.mean(teacher_density)),
        }
    else:
        print('Generating raw motif-scale teacher...')
        _, teacher_raw, teacher_density, teacher_metrics = run_teacher(
            golden_mod=golden_mod, max_iterations=max_iterations)
        teacher_metrics['reused'] = False
        persist_teacher_fields(persist_dir, teacher_raw, teacher_density)
        (persist_dir / 'summary.json').write_text(
            json.dumps(teacher_metrics, indent=2) + '\n')

    scaffold = extract_scaffold(teacher_raw, layout, scale_fracs)
    np.save(output_dir / 'scaffold.npy', np.ascontiguousarray(scaffold))
    reference = Image.open(GOLDEN_FINAL_IMAGE_PATH).convert('L')
    _ink_black(scaffold, reference.size).save(output_dir / 'scaffold.png')
    print(f'Wrote scaffold preview to {output_dir / "scaffold.png"}')

    if scaffold_only:
        summary = {
            'scaffold_only': True,
            'layout': {
                'threshold': layout.threshold,
                'envelope_sigma_frac': layout.envelope_sigma_frac,
                'combine': layout.combine,
                'weight': layout.weight,
                'weight_end': layout.weight_end,
                'scale_fracs': list(scale_fracs),
            },
            'teacher': teacher_metrics,
            'scaffold_mean': float(np.mean(scaffold)),
        }
        (output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n')
        return summary

    print('Running layout student (raw CLIP + scaffold mass prior)...')
    experiment, model, ds = run_student(
        golden_mod=golden_mod,
        teacher_raw=teacher_raw,
        max_iterations=max_iterations,
    )
    student_raw = np.ascontiguousarray(
        ds['final_design_raw'].values, dtype=np.float32)
    student_density = _physical_density(model)
    load_sites = load_site_mask(
        model.env.args['forces'],
        nely=int(model.env.args['nely']),
        nelx=int(model.env.args['nelx']),
    )
    allowed = np.maximum(scaffold, load_sites)
    paths = save_layout_look(
        output_dir=output_dir,
        teacher_raw=teacher_raw,
        teacher_density=teacher_density,
        scaffold=scaffold,
        student_raw=student_raw,
        student_density=student_density,
    )
    density_t = torch.as_tensor(student_density, dtype=torch.float32)
    occupancy_t = torch.as_tensor(scaffold, dtype=torch.float32)
    load_sites_t = torch.as_tensor(load_sites, dtype=torch.float32)
    student_metrics = _metrics_from_dataset(ds, golden_mod, student_raw)
    summary = {
        'clip_prompt': GOLDEN.prompt,
        'occupancy': False,
        'layout': {
            'enabled': True,
            'threshold': layout.threshold,
            'envelope_sigma_frac': layout.envelope_sigma_frac,
            'combine': layout.combine,
            'weight': layout.weight,
            'weight_end': layout.weight_end,
            'init_from_teacher': layout.init_from_teacher,
            'scale_fracs': list(scale_fracs),
        },
        'teacher': teacher_metrics,
        'teacher_baseline_compliance': TEACHER_BASELINE_COMPLIANCE,
        'mass_on_scaffold': mass_fraction_on_occupancy(
            student_density, scaffold),
        'mass_on_allowed': mass_fraction_on_occupancy(
            student_density, allowed),
        'spatial_mass_loss': float(sketch_mass_prior_loss(
            density_t, occupancy_t, load_sites=load_sites_t)),
        'load_site_frac': float(load_sites.mean()),
        'scaffold_mean': float(np.mean(scaffold)),
        'mean_physical_density': float(np.mean(student_density)),
        **student_metrics,
        'gate': {
            'mean_density_target': TEACHER_BASELINE_MEAN_DENSITY,
            'compliance_not_worse_than': TEACHER_BASELINE_COMPLIANCE,
            'compliance_held': (
                student_metrics['compliance'] <= TEACHER_BASELINE_COMPLIANCE),
            'mean_density_held': abs(
                float(np.mean(student_density))
                - TEACHER_BASELINE_MEAN_DENSITY) <= 0.02,
        },
        'paths': {key: str(path) for key, path in paths.items()},
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    persist_teacher_fields(persist_dir, teacher_raw, teacher_density)
    np.save(output_dir / 'student_raw.npy', student_raw)
    np.save(output_dir / 'student_physical_density.npy', student_density)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='student result directory (does not overwrite prior A/B looks)',
    )
    parser.add_argument(
        '--teacher-dir',
        type=Path,
        default=None,
        help=(
            'reuse final_design_raw.npy and physical_density.npy from this '
            'directory when both exist; otherwise generate a teacher'),
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='cap gradient steps for a shortened teacher/student look',
    )
    parser.add_argument(
        '--scaffold-only',
        action='store_true',
        help='extract and save the scaffold, then exit without the student',
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    teacher_dir = args.teacher_dir
    if teacher_dir is not None and not teacher_dir.is_absolute():
        teacher_dir = REPO_ROOT / teacher_dir
    elif teacher_dir is None and load_teacher_fields(PRIOR_TEACHER_DIR) is not None:
        teacher_dir = PRIOR_TEACHER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_distillation(
        output_dir=output_dir,
        teacher_dir=teacher_dir,
        max_iterations=args.max_iterations,
        scaffold_only=args.scaffold_only,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
