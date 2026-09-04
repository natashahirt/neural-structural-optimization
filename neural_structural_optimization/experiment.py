"""Typed experiment configuration for comparable research runs.

Stage 5 of the hardfork consolidation: a frozen config object that names every
knob a run actually uses, without loading CLIP or building a model. The Venice
``250214`` golden knobs live here so ``--print-config`` and the golden script
cannot drift apart.

This module is deliberately free of CLIP and kornia. ``resolve()`` and
``--print-config`` must stay that way: constructing ``CLIPLoss`` downloads
weights and the whole point of a config-only path is to inspect a run before
paying that cost. torch may still load as a side-effect of importing
``StructuralParams`` (``structural/__init__.py`` pulls in ``api``), which is
left eager so CHOLMOD never races a late torch OpenMP import.
``run()`` lazy-imports the golden script when a Venice-compat adaptive-pixel
preset is actually executed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from neural_structural_optimization.structural.problems import (
    DISCRETIZATION_FIELDS,
    PHYSICS_DISCRETIZATION_DEFAULTS,
    StructuralParams,
    resolve_discretization_value,
)

# Repo-relative path of the Venice seed image. The golden script turns this
# into an absolute Path next to its own copied asset; the config stores the
# relative form so JSON manifests stay portable.
VENICE_250214_IMAGE = (
    'script/resources/input_images/dSketches/thick_outer_lins.png')

# Physics hardcodes this in `specified_task`; the config keeps the value so a
# mismatch is a resolve-time error rather than a silent wrong stiffness.
PHYSICS_PENAL = 3.0

KNOWN_MODEL_KINDS = ('adaptive_pixel', 'pixel', 'cnn')
KNOWN_OPTIMIZER_KINDS = ('adaptive_adam', 'adam', 'lbfgs')


# ---------------------------------------------------------------------------
# Venice golden knobs. Moved here from script/venice_golden_250214.py so a
# config-only import does not load that script's CLIP/torch builders. Values
# are the reference run's log, not Venice's drifted checked-in main.py.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VeniceGoldenConfig:
    """Every knob of the ``250214_skeleton_loss_test_balanced_dynamic`` run.

    Attributes are grouped by the object that consumes them: the structural
    problem, the model's resolution schedule, the CLIP loss, the loss algebra
    and the optimizer.
    """

    problem_name: str = 'multistory_building'
    width: int = 128
    height: int = 256
    density: float = 0.3
    interval: int = 64
    filter_width: float = 2.0
    penal: float = 3.0

    resize_num: int = 2
    resize_scale: int = 2

    clip_model_name: str = 'ViT-B/32'
    # Venice loads a second ResNet CLIP for a geometric loss that is off in
    # this run. RN50 stands in for Venice's RN101 because it is smaller; the
    # Venice CLIP path never touches the ResNet trunk.
    clip_rn_model_name: str = 'RN50'
    prompt: str = 'skeletons'
    num_augs: int = 32
    clip_resize_short_side: int = 512

    clip_alpha: float = 10.0
    compliance_weight: float = 1.0

    lr: float = 0.2
    max_iterations: int = 200
    resize_threshold: float = 0.5
    max_resize_iteration: int = 50
    convergence_threshold: float = 0.05

    seed: int = 12
    device: str = 'cpu'
    invert_image: bool = True


GOLDEN = VeniceGoldenConfig()

# Cheap variant for the default test suite: numbers are not comparable to the
# reference log. It exists to prove the preset, image seeding, schedule and
# term breakdown still hold together.
SMOKE = VeniceGoldenConfig(
    width=32,
    height=64,
    interval=16,
    num_augs=4,
    clip_resize_short_side=128,
    max_iterations=4,
)


# ---------------------------------------------------------------------------
# ExperimentConfig sections
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProblemConfig:
    """Structural problem at its full (final) resolution."""

    problem_name: str = 'cantilever_beam_full'
    width: int = 60
    height: int = 60
    density: float = 0.5
    interval: int = 16
    filter_width: Union[float, str, None] = None
    rmin: Union[float, str, None] = None
    beta: Union[float, str, None] = None
    heavyside: Optional[bool] = None
    eta: Optional[float] = None
    penal: float = PHYSICS_PENAL
    fix_right_wall: bool = True


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Which parameterization owns the design, and its resolution schedule."""

    kind: str = 'adaptive_pixel'
    resize_num: int = 0
    resize_scale: int = 2
    venice_loss_algebra: bool = False


@dataclasses.dataclass(frozen=True)
class ClipConfig:
    """Semantic guidance. ``enabled`` False is a structure-only run."""

    enabled: bool = False
    venice_compat: bool = False
    clip_model_name: str = 'ViT-B/32'
    clip_rn_model_name: str = 'RN50'
    prompts: tuple[str, ...] = ()
    num_augs: int = 24
    clip_resize_short_side: int = 512


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Who steps the design, and the loss-algebra coefficients they pass."""

    kind: str = 'adaptive_adam'
    lr: float = 0.01
    max_iterations: int = 100
    clip_alpha: float = 10.0
    compliance_weight: float = 1.0
    resize_threshold: float = 0.5
    max_resize_iteration: int = 50
    convergence_threshold: float = 0.05


@dataclasses.dataclass(frozen=True)
class InitConfig:
    """Optional image seed, in [0, 1] pixel space."""

    image: Optional[str] = None
    invert_image: bool = True


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Seed, device, and where artifacts land."""

    seed: int = 0
    device: str = 'cpu'
    output_dir: str = 'script/test_results_pytorch'


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment. Sections compose existing objects; they do not
    invent a second density chain or a second loss algebra.
    """

    name: str = 'unnamed'
    problem: ProblemConfig = dataclasses.field(default_factory=ProblemConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    clip: ClipConfig = dataclasses.field(default_factory=ClipConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    init: InitConfig = dataclasses.field(default_factory=InitConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)

    # -- construction from the golden knobs ---------------------------------

    @classmethod
    def from_venice_golden(
        cls,
        golden: VeniceGoldenConfig = GOLDEN,
        *,
        name: str = 'venice_250214',
    ) -> 'ExperimentConfig':
        """Lift a ``VeniceGoldenConfig`` into the typed experiment object.

        The mapping is 1-1 on knobs. Extra ExperimentConfig fields (model kind,
        venice flags, image path) are the Stage 5 defaults for this preset, not
        new research choices.
        """
        return cls(
            name=name,
            problem=ProblemConfig(
                problem_name=golden.problem_name,
                width=golden.width,
                height=golden.height,
                density=golden.density,
                interval=golden.interval,
                filter_width=golden.filter_width,
                penal=golden.penal,
            ),
            model=ModelConfig(
                kind='adaptive_pixel',
                resize_num=golden.resize_num,
                resize_scale=golden.resize_scale,
                venice_loss_algebra=True,
            ),
            clip=ClipConfig(
                enabled=True,
                venice_compat=True,
                clip_model_name=golden.clip_model_name,
                clip_rn_model_name=golden.clip_rn_model_name,
                prompts=(golden.prompt,),
                num_augs=golden.num_augs,
                clip_resize_short_side=golden.clip_resize_short_side,
            ),
            optimizer=OptimizerConfig(
                kind='adaptive_adam',
                lr=golden.lr,
                max_iterations=golden.max_iterations,
                clip_alpha=golden.clip_alpha,
                compliance_weight=golden.compliance_weight,
                resize_threshold=golden.resize_threshold,
                max_resize_iteration=golden.max_resize_iteration,
                convergence_threshold=golden.convergence_threshold,
            ),
            init=InitConfig(
                image=VENICE_250214_IMAGE,
                invert_image=golden.invert_image,
            ),
            run=RunConfig(
                seed=golden.seed,
                device=golden.device,
            ),
        )

    def to_venice_golden(self) -> VeniceGoldenConfig:
        """Project this config back onto the golden dataclass.

        Used by the Stage 5 gate (round-trip equals ``GOLDEN``) and by ``run()``
        so the existing golden builders stay the execution path.
        """
        if len(self.clip.prompts) != 1:
            raise ValueError(
                'VeniceGoldenConfig carries a single prompt; this experiment '
                f'has {len(self.clip.prompts)}: {self.clip.prompts!r}.')
        return VeniceGoldenConfig(
            problem_name=self.problem.problem_name,
            width=self.problem.width,
            height=self.problem.height,
            density=self.problem.density,
            interval=self.problem.interval,
            filter_width=_require_float(
                self.problem.filter_width, 'problem.filter_width'),
            penal=self.problem.penal,
            resize_num=self.model.resize_num,
            resize_scale=self.model.resize_scale,
            clip_model_name=self.clip.clip_model_name,
            clip_rn_model_name=self.clip.clip_rn_model_name,
            prompt=self.clip.prompts[0],
            num_augs=self.clip.num_augs,
            clip_resize_short_side=self.clip.clip_resize_short_side,
            clip_alpha=self.optimizer.clip_alpha,
            compliance_weight=self.optimizer.compliance_weight,
            lr=self.optimizer.lr,
            max_iterations=self.optimizer.max_iterations,
            resize_threshold=self.optimizer.resize_threshold,
            max_resize_iteration=self.optimizer.max_resize_iteration,
            convergence_threshold=self.optimizer.convergence_threshold,
            seed=self.run.seed,
            device=self.run.device,
            invert_image=self.init.invert_image,
        )

    # -- StructuralParams / effective settings ------------------------------

    def structural_params(self) -> StructuralParams:
        """Build the ``StructuralParams`` the golden builders would build."""
        kwargs: dict[str, Any] = dict(
            problem_name=self.problem.problem_name,
            width=self.problem.width,
            height=self.problem.height,
            density=self.problem.density,
            interval=self.problem.interval,
            fix_right_wall=self.problem.fix_right_wall,
        )
        for name in ('filter_width', 'rmin', 'beta', 'heavyside', 'eta'):
            value = getattr(self.problem, name)
            if value is not None:
                kwargs[name] = value
        return StructuralParams(**kwargs)

    def resolve(self) -> dict[str, Any]:
        """Effective settings, JSON-friendly, without loading CLIP or a model.

        Discretization strings such as ``'linear'`` are resolved the same way
        ``StructuralParams.get_problem`` would. The coarse-start grid is the
        authored size divided by ``resize_scale ** resize_num``. ``penal`` is
        checked against the physics constant because ``specified_task`` will
        ignore any other value.
        """
        self.validate()
        params = self.structural_params()
        effective = _resolved_discretization(params)
        divisor = int(self.model.resize_scale) ** int(self.model.resize_num)
        payload = self.to_dict()
        payload['effective'] = {
            **effective,
            'penal': PHYSICS_PENAL,
            'coarse_width': params.width // divisor,
            'coarse_height': params.height // divisor,
            'coarse_interval': max(1, params.interval // divisor),
            'schedule_divisor': divisor,
        }
        return payload

    def validate(self) -> None:
        """Raise ``ValueError`` if this config cannot be executed as authored."""
        if self.model.kind not in KNOWN_MODEL_KINDS:
            raise ValueError(
                f'unknown model.kind {self.model.kind!r}; '
                f'expected one of {KNOWN_MODEL_KINDS}')
        if self.optimizer.kind not in KNOWN_OPTIMIZER_KINDS:
            raise ValueError(
                f'unknown optimizer.kind {self.optimizer.kind!r}; '
                f'expected one of {KNOWN_OPTIMIZER_KINDS}')
        if self.problem.penal != PHYSICS_PENAL:
            raise ValueError(
                f'problem.penal is {self.problem.penal}, but specified_task '
                f'hardcodes penal={PHYSICS_PENAL}; the authored value would '
                'be ignored.')
        divisor = int(self.model.resize_scale) ** int(self.model.resize_num)
        if divisor and (self.problem.width % divisor
                        or self.problem.height % divisor):
            raise ValueError(
                f'{self.problem.width}x{self.problem.height} is not divisible '
                f'by resize_scale ** resize_num = {divisor}.')
        if self.clip.venice_compat and not self.model.venice_loss_algebra:
            raise ValueError(
                'clip.venice_compat requires model.venice_loss_algebra: the '
                'legacy CLIP path and the legacy total-loss algebra are one '
                'preset, not two independent flags.')
        if self.clip.enabled and self.clip.venice_compat and not self.clip.prompts:
            raise ValueError(
                'Venice CLIP needs at least one prompt; clip.prompts is empty.')

    # -- serialization / overrides ------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Nested dict with tuples converted to lists for JSON."""
        return _jsonify(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'ExperimentConfig':
        """Rebuild from ``to_dict`` output or a JSON manifest."""
        if not isinstance(data, Mapping):
            raise TypeError(
                f'ExperimentConfig.from_dict expected a mapping, got '
                f'{type(data).__name__}')
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f'unknown ExperimentConfig field(s) {sorted(unknown)}; '
                f'known fields are {sorted(known)}')
        kwargs: dict[str, Any] = {}
        if 'name' in data:
            kwargs['name'] = data['name']
        section_types = {
            'problem': ProblemConfig,
            'model': ModelConfig,
            'clip': ClipConfig,
            'optimizer': OptimizerConfig,
            'init': InitConfig,
            'run': RunConfig,
        }
        for section, section_cls in section_types.items():
            if section in data:
                kwargs[section] = _section_from_dict(section_cls, data[section])
        return cls(**kwargs)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + '\n'

    @classmethod
    def from_json(cls, text: str) -> 'ExperimentConfig':
        return cls.from_dict(json.loads(text))

    def with_overrides(
        self,
        overlay: Optional[Mapping[str, Any]] = None,
        **dotted: Any,
    ) -> 'ExperimentConfig':
        """Return a copy with nested or dotted-key overrides.

        The original is untouched. Nested dicts merge; dotted keys such as
        ``optimizer.lr=0.1`` set a single leaf. Unknown sections or fields
        raise ``ValueError``.
        """
        data = dataclasses.asdict(self)
        if overlay:
            data = _deep_merge(data, dict(overlay))
        for key, value in dotted.items():
            _set_dotted(data, key, value)
        return type(self).from_dict(_jsonify(data))


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def venice_250214() -> ExperimentConfig:
    """The Stage 3 golden run, as a typed experiment."""
    return ExperimentConfig.from_venice_golden(GOLDEN, name='venice_250214')


def venice_250214_smoke() -> ExperimentConfig:
    """The cheap smoke variant of the golden run (numbers are not comparable)."""
    return ExperimentConfig.from_venice_golden(
        SMOKE, name='venice_250214_smoke')


PRESETS = {
    'venice_250214': venice_250214,
    'venice_250214_smoke': venice_250214_smoke,
}


def preset(name: str) -> ExperimentConfig:
    """Look up a named preset. Raises ``KeyError`` with the known list."""
    try:
        factory = PRESETS[name]
    except KeyError as exc:
        known = ', '.join(sorted(PRESETS))
        raise KeyError(
            f'unknown preset {name!r}; known presets: {known}') from exc
    return factory()


# ---------------------------------------------------------------------------
# Execution (lazy CLIP)
# ---------------------------------------------------------------------------

def run(config: ExperimentConfig, **kwargs: Any):
    """Execute a Venice-compat adaptive-pixel experiment.

    Delegates to ``script/venice_golden_250214.py`` so Stage 5 cannot fork a
    second golden path. Other model/optimizer kinds are not wired yet.
    """
    config.validate()
    if (config.model.kind != 'adaptive_pixel'
            or not config.clip.venice_compat
            or not config.model.venice_loss_algebra
            or config.optimizer.kind != 'adaptive_adam'):
        raise NotImplementedError(
            'Stage 5 runner only executes Venice-compat AdaptivePixel + '
            f'AdaptiveAdam presets; got model={config.model.kind!r}, '
            f'optimizer={config.optimizer.kind!r}, '
            f'venice_compat={config.clip.venice_compat}.')
    golden_mod = _load_golden_script()
    return golden_mod.run(config.to_venice_golden(), **kwargs)


def _load_golden_script():
    """Import the golden script from ``script/``, which is not a package."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / 'script' / 'venice_golden_250214.py'
    spec = importlib.util.spec_from_file_location('venice_golden_250214', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """``--print-config`` never loads CLIP. ``--run`` does, lazily."""
    parser = argparse.ArgumentParser(
        prog='python -m neural_structural_optimization.experiment',
        description=(
            'Typed experiment config. --print-config writes the resolved '
            'settings without loading CLIP. --run executes a Venice-compat '
            'preset through the existing golden builders.'),
    )
    parser.add_argument(
        '--print-config',
        metavar='PRESET',
        nargs='?',
        const='venice_250214',
        help='print resolved JSON for PRESET (default: venice_250214)',
    )
    parser.add_argument(
        '--run',
        metavar='PRESET',
        dest='run_preset',
        help='execute PRESET (loads CLIP)',
    )
    parser.add_argument(
        '--json',
        metavar='PATH',
        dest='json_path',
        help='load an ExperimentConfig manifest instead of a named preset',
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='dotted override, repeatable (e.g. optimizer.lr=0.1)',
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='print known preset names and exit',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_presets:
        for name in sorted(PRESETS):
            print(name)
        return 0

    if args.print_config is None and args.run_preset is None and args.json_path is None:
        parser.print_help()
        print('\nPresets:', ', '.join(sorted(PRESETS)))
        return 2

    try:
        config = _config_from_cli(args)
        for item in args.override:
            key, value = _split_override(item)
            config = config.with_overrides(**{key: value})
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    if args.run_preset is not None:
        run(config)
        return 0

    print(json.dumps(config.resolve(), indent=2, sort_keys=True))
    return 0


def _config_from_cli(args: argparse.Namespace) -> ExperimentConfig:
    if args.json_path:
        return ExperimentConfig.from_json(Path(args.json_path).read_text())
    name = args.run_preset or args.print_config or 'venice_250214'
    return preset(name)


def main() -> int:
    return cli_main()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_float(value: Union[float, str, None], name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f'{name} must be a number to project onto VeniceGoldenConfig; '
            f'got {value!r}.')
    return float(value)


def _resolved_discretization(params: StructuralParams) -> dict[str, Any]:
    """Concrete filter/projection values physics would see, without FEA."""
    rmin = params.rmin
    if rmin is None:
        rmin = PHYSICS_DISCRETIZATION_DEFAULTS['rmin']
    elif isinstance(rmin, str):
        raise ValueError(f'rmin schedule strings are not supported; got {rmin!r}')
    rmin = float(rmin)
    out: dict[str, Any] = {}
    for name in DISCRETIZATION_FIELDS:
        raw = getattr(params, name)
        if raw is None:
            out[name] = PHYSICS_DISCRETIZATION_DEFAULTS[name]
        else:
            out[name] = resolve_discretization_value(
                name, raw, rmin=rmin,
                nelx=params.width, nely=params.height)
    out['rmin'] = rmin
    return out


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    return value


def _section_from_dict(section_cls, data: Mapping[str, Any]):
    if not isinstance(data, Mapping):
        raise TypeError(
            f'{section_cls.__name__} expected a mapping, got '
            f'{type(data).__name__}')
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f'unknown {section_cls.__name__} field(s) {sorted(unknown)}; '
            f'known fields are {sorted(known)}')
    kwargs = dict(data)
    if section_cls is ClipConfig and 'prompts' in kwargs:
        kwargs['prompts'] = tuple(kwargs['prompts'])
    return section_cls(**kwargs)


def _deep_merge(base: dict, overlay: Mapping[str, Any]) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if (key in out and isinstance(out[key], dict)
                and isinstance(value, Mapping)):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split('.')
    if len(parts) < 2:
        if parts[0] == 'name':
            data['name'] = value
            return
        raise ValueError(
            f'override {dotted!r} is not a dotted section.field key')
    cur = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            raise ValueError(
                f'cannot override {dotted!r}: {part!r} is not a section')
        cur = cur[part]
    leaf = parts[-1]
    if leaf not in cur:
        raise ValueError(
            f'cannot override {dotted!r}: unknown field {leaf!r}')
    cur[leaf] = value


def _split_override(item: str) -> tuple[str, Any]:
    if '=' not in item:
        raise ValueError(
            f'override {item!r} must be KEY=VALUE (e.g. optimizer.lr=0.1)')
    key, raw = item.split('=', 1)
    key = key.strip()
    raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


if __name__ == '__main__':
    raise SystemExit(main())
