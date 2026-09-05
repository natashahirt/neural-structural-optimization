"""Base model class for neural structural optimization."""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_structural_optimization.structural.problems import (
    StructuralParams,
    resolve_analysis_filter_width,
)
from .loss_structural import StructuralLoss, PhysicalDensity
from .loss_clip import CLIPLoss
from .loss_sketch import (
    load_site_mask,
    sketch_mass_prior_loss,
    sketch_motif_loss,
    sketch_patch_vocabulary_loss,
)
from .config import DEFAULT_MAX_ANALYSIS_DIM
from .utils import set_random_seed
from neural_structural_optimization.structural import api as topo_api


# ---------------------------------------------------------------------------
# Venice legacy-compatibility seam
#
# The default total loss couples CLIP to compliance through a *detached* weight
# and has no unweighted term. The legacy Venice run does neither, so the two
# algebras live side by side and are selected per model instance. See
# `venice_compat_total_loss` for the formula and `Model.get_total_loss` for the
# single dispatch point.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VeniceLossAlgebra:
    """Coefficients of the legacy Venice total-loss algebra.

    Attributes:
        clip_alpha: Multiplier turning the compliance loss into the CLIP weight.
        compliance_weight: Multiplier on the raw structural loss.

    Defaults are the values of the reference run.
    """

    clip_alpha: float = 10.0
    compliance_weight: float = 1.0


VENICE_LOSS_ALGEBRA = VeniceLossAlgebra()


class VeniceLossTerms(NamedTuple):
    """Per-step loss breakdown, named after the legacy log fields."""

    total_loss: torch.Tensor
    compliance_loss: torch.Tensor
    clip_loss: torch.Tensor
    clip_loss_raw: torch.Tensor
    clip_weight: torch.Tensor


def _coerce_venice_loss_algebra(value) -> Optional[VeniceLossAlgebra]:
    """Normalize a ``venice_compat`` argument to an algebra or None.

    Accepts None/False (default algebra), True (the reference coefficients), or
    an explicit :class:`VeniceLossAlgebra`, so the seam can be driven straight
    from a boolean configuration flag.
    """
    if value is None or value is False:
        return None
    if value is True:
        return VENICE_LOSS_ALGEBRA
    if isinstance(value, VeniceLossAlgebra):
        return value
    raise TypeError(
        "venice_compat must be None, a bool, or a VeniceLossAlgebra, "
        f"got {type(value).__name__}."
    )


def venice_compat_total_loss(
    structural_loss: torch.Tensor,
    clip_loss_raw: torch.Tensor,
    *,
    clip_alpha: float,
    compliance_weight: float,
) -> VeniceLossTerms:
    """Combine the losses exactly as the legacy Venice ``LossObject`` does::

        compliance_loss = structural_loss * compliance_weight
        clip_weight     = compliance_loss * clip_alpha
        clip_loss       = clip_loss_raw * clip_weight
        total_loss      = compliance_loss + clip_loss + clip_loss_raw

    Two properties are load-bearing and deliberate, not oversights:

    * ``clip_weight`` is **not** detached. Compliance therefore receives
      gradient through the CLIP term as well as directly, which is a real part
      of the legacy dynamics.
    * The raw, unweighted CLIP loss is added *on top of* the weighted one,
      because Venice sums every populated field of its loss dictionary.

    Both were confirmed arithmetically against the reference log: a compliance
    of 73.99691670938864 at ``clip_alpha=10`` gives a weight of 739.97, and
    ``283.3703603894398 / 0.3829488754272461`` is exactly 740.0; the three terms
    sum to the logged total of 357.75022597425567.

    Args:
        structural_loss: Unweighted compliance from the physics solve.
        clip_loss_raw: Unweighted CLIP loss (see `loss_clip.VeniceClipPath`).
        clip_alpha: Compliance-to-CLIP-weight multiplier.
        compliance_weight: Multiplier on the structural loss.

    Returns:
        The full :class:`VeniceLossTerms` breakdown.
    """
    compliance_loss = structural_loss * float(compliance_weight)
    clip_weight = compliance_loss * float(clip_alpha)
    clip_loss = clip_loss_raw * clip_weight
    total_loss = compliance_loss + clip_loss + clip_loss_raw
    return VeniceLossTerms(
        total_loss=total_loss,
        compliance_loss=compliance_loss,
        clip_loss=clip_loss,
        clip_loss_raw=clip_loss_raw,
        clip_weight=clip_weight,
    )


class Model(nn.Module):
    """Base model class for structural optimization."""
    
    def __init__(
        self, 
        structural_params: Optional[StructuralParams | dict] = None, 
        clip_loss: Optional[object] = None, 
        seed: Optional[int] = None, 
        args: Optional[dict] = None,
        venice_compat: Optional[VeniceLossAlgebra | bool] = None,
    ):
        super().__init__()

        # Legacy loss algebra; None keeps the default detached coupling.
        # Subclasses that do not forward this kwarg can use
        # `enable_venice_compat_loss()` after construction instead.
        self.venice_loss_algebra = _coerce_venice_loss_algebra(venice_compat)
        
        # Handle problem parameters
        if structural_params is not None:
            if isinstance(structural_params, dict):
                self.structural_params = StructuralParams(**structural_params)
            else:
                self.structural_params = structural_params

            if args is None:
                args = self._build_physics_args()
        else:
            self.structural_params = None

        # Set random seed
        set_random_seed(seed)
        self.seed = seed
        
        # Initialize environment
        self.env = topo_api.Environment(args)
        self.args = args
        
        # Initialize mask tensor once
        self.mask = torch.as_tensor(self.args['mask'], dtype=torch.float64)
        
        # Analysis settings
        self.analysis_factor = 1
        self.analysis_env = self.env

        if hasattr(self, 'z'):
            self.device = self.z.device
        else: 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        object.__setattr__(self, "clip_loss", None)  # placeholder
        self.clip_weight_default = 1.0
        # Stage 6 mass prior. None/0 keeps every loss path bit-identical to a
        # no-sketch run (no extra PhysicalDensity eval, no extra term).
        self.sketch_occupancy_full = None
        self.sketch_weight = 0.0
        self.sketch_weight_start = 0.0
        self.sketch_weight_end = None
        self.sketch_motif_weight = 0.0
        self.sketch_motif_weight_peak = 0.0
        self.sketch_motif_weight_end = None
        self.sketch_motif_scales = (1, 2, 4)
        self.sketch_patch_weight = 0.0
        self.sketch_patch_weight_peak = 0.0
        self.sketch_patch_weight_end = None
        self.sketch_patch_sizes = (7, 15)
        self.sketch_patch_stride = 2
        if clip_loss is not None:
            clip_loss.clip_model = (
                clip_loss.clip_model.to(self.device).eval().requires_grad_(False)
            )
            if self.device.type == "cpu":
                clip_loss.clip_model = clip_loss.clip_model.float()
            if hasattr(clip_loss, "device"):
                clip_loss.device = self.device
            if 'clip_loss' in self._modules:
                del self._modules['clip_loss']
            object.__setattr__(self, "clip_loss", clip_loss)
            self.clip_R = 0.9

    def forward(self) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the design grid."""
        return (1, self.env.args['nely'], self.env.args['nelx'])

    @property
    def class_name(self) -> str:
        """Get the name of the model class."""
        return self.__class__.__name__

    def _get_mask_3d(self, H: int, W: int) -> torch.Tensor:
        """Get mask as 3D tensor with shape (1, H, W)."""
        # Get device and dtype from z if available, otherwise use defaults
        if hasattr(self, 'z'):
            device = self.z.device
            dtype = self.z.dtype
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32
            
        mask = torch.as_tensor(
            self.args['mask'], 
            device=device, 
            dtype=dtype
        )
        
        if mask.ndim == 0:
            mask = torch.ones(
                (1, H, W), 
                dtype=dtype, 
                device=device
            ) * mask
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return mask

    def _build_physics_args(self) -> dict:
        """Build physics args with discretization schedules resolved to concrete values."""
        return topo_api.specified_task(self.structural_params.get_problem())

    def _refresh_physics_environment(self) -> None:
        """Rebuild env/args after structural or discretization parameters change."""
        new_args = self._build_physics_args()
        self.env = topo_api.Environment(new_args)
        self.args = new_args

    def _update_structural_params(self, scale: Optional[float] = None) -> None:
        """Update structural parameters, typically for upsampling."""
        new_params = {}

        if scale is not None:
            new_params['width'] = int(self.structural_params.width * scale)
            new_params['height'] = int(self.structural_params.height * scale)
            
            # Only scale numeric parameters
            if isinstance(self.structural_params.rmin, (int, float)):
                new_params['rmin'] = self.structural_params.rmin * scale
            if isinstance(self.structural_params.filter_width, (int, float)):
                new_params['filter_width'] = self.structural_params.filter_width * scale
            
            # Beta usually doesn't scale with resolution unless explicitly requested
            # but we'll preserve its value/schedule
            new_params['beta'] = self.structural_params.beta
            new_params['heavyside'] = self.structural_params.heavyside
            new_params['eta'] = self.structural_params.eta

        self.structural_params = self.structural_params.copy(**new_params)
        self._refresh_physics_environment()
        
        # Update mask for new structural parameters
        self.mask = torch.as_tensor(self.args['mask'], dtype=torch.float64)

    def _set_analysis_factor(self, max_dim: int = DEFAULT_MAX_ANALYSIS_DIM, reset: bool = False) -> None:
        """Set analysis factor for downsampling during physics computation.

        The analysis grid divides both the grid and the filter radius by the
        analysis factor, so the derived radius can fall below the degeneracy
        threshold for a configuration that is perfectly valid on the design
        grid. This runs after a stage has finished training (`PixelModel.
        upsample` calls it), so `resolve_analysis_filter_width` clamps such a
        radius with a warning instead of raising and discarding the stage.
        """
        _, H, W = self.shape
        f = max(1, max((H + max_dim - 1) // max_dim, (W + max_dim - 1) // max_dim))
        self.analysis_factor = f

        if f == 1 or reset:
            self.analysis_env = self.env
            self.analysis_factor = 1
            return

        # Get analysis environment
        analysis_dict = {
            'width': int(round(W / f)),
            'height': int(round(H / f)),
        }

        # Only scale if numeric
        if isinstance(self.structural_params.rmin, (int, float)):
            analysis_dict['rmin'] = self.structural_params.rmin / f
        else:
            analysis_dict['rmin'] = self.structural_params.rmin

        if isinstance(self.structural_params.filter_width, (int, float)):
            scaled_filter_width = self.structural_params.filter_width / f
        else:
            scaled_filter_width = self.structural_params.filter_width

        # Resolve to a concrete radius here so the clamp applies to the value
        # physics will actually use, rather than to an unresolved schedule.
        analysis_dict['filter_width'] = resolve_analysis_filter_width(
            scaled_filter_width,
            rmin=analysis_dict['rmin'],
            nelx=analysis_dict['width'],
            nely=analysis_dict['height'],
        )

        analysis_params = self.structural_params.copy(**analysis_dict)
        analysis_args = topo_api.specified_task(analysis_params.get_problem())
        analysis_args['volfrac'] = self.args.get('volfrac', analysis_args.get('volfrac', 0.5))

        self.analysis_env = topo_api.Environment(analysis_args)

    def _downfactor_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Downsample logits for analysis computation."""
        _, H, W = self.shape
        f = self.analysis_factor

        # Get "element densities" from analysis grid
        mask = self._get_mask_3d(H, W)

        z_4d = (z * mask).unsqueeze(0)  # (N=1, C=1, H, W)
        m_4d = mask.unsqueeze(0)

        num = F.avg_pool2d(z_4d, kernel_size=f, stride=f, ceil_mode=True)
        den = F.avg_pool2d(m_4d, kernel_size=f, stride=f, ceil_mode=True).clamp_min(1e-12)

        z_coarse = (num / den).squeeze(0) 
        return z_coarse

    def get_structural_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute physics-based structural loss."""
        if not hasattr(self, 'analysis_factor'):
            self._set_analysis_factor()
            
        env = self.env
        z = logits
        if getattr(self, 'analysis_factor', 1) != 1:
            z = self._downfactor_logits(logits)
            env = self.analysis_env

        # Use NumPy/HIPS-autograd bridge (faster/stable on CPU)
        return StructuralLoss.apply(z, env).mean()

    def get_physical_density(self, logits: torch.Tensor) -> torch.Tensor:
        """The canonical density: filtered, volume-constrained, same as render.

        Differentiable through a HIPS VJP. Evaluated on `self.env` (full
        resolution), not `analysis_env`. Venice CLIP does not use this.
        """
        return PhysicalDensity.apply(logits, self.env)

    def _clip_sees_raw_design(self) -> bool:
        """Venice CLIP (and the Venice algebra's stub) consume raw z."""
        if getattr(self.clip_loss, 'venice_path', None) is not None:
            return True
        return self.venice_loss_algebra is not None

    def get_semantic_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute clip-based semantic loss on the canonical density.

        Default CLIP sees `get_physical_density(logits)`, not `sigmoid(z)`.
        The Venice preset still passes raw logits, matching the golden run.
        """
        if self.clip_loss is None:
            return logits.new_tensor(0.0)
        if self._clip_sees_raw_design():
            return self.clip_loss(logits)
        return self.clip_loss(self.get_physical_density(logits))

    def enable_sketch_prior(
        self,
        occupancy: torch.Tensor | np.ndarray,
        weight: float = 1.0,
        weight_end: Optional[float] = None,
        motif_weight: float = 0.0,
        motif_weight_end: Optional[float] = None,
        motif_scales: Tuple[int, ...] = (1, 2, 4),
        patch_weight: float = 0.0,
        patch_weight_end: Optional[float] = None,
        patch_sizes: Tuple[int, ...] = (7, 15),
        patch_stride: int = 2,
    ) -> "Model":
        """Pull the canonical physical density toward a sketch occupancy map.

        Occupancy is stored at whatever grid it was loaded on (the full
        problem size). ``get_sketch_loss`` resamples it onto the current
        logits, so a coarse AdaptivePixel stage sees a downsampled prior
        rather than a missing one.

        Args:
            occupancy: ``(H, W)`` or ``(1, H, W)`` values in ``[0, 1]``. Ink
                is 1. Not a Parameter: the sketch does not train.
            weight: Multiplier on :func:`sketch_mass_prior_loss` at the
                start of a run (coarse AdaptivePixel stage, or step 0).
                0 disables the term without clearing the map.
            weight_end: If set, the multiplier anneals toward this value
                (finest stage, or last step). ``None`` keeps ``weight``
                constant.
            motif_weight: Peak multiplier on the translation-invariant
                sketch motif loss. AdaptivePixel keeps it off at the coarse
                stage and reaches this value after the first upsample.
            motif_weight_end: Motif multiplier at the finest stage. ``None``
                keeps ``motif_weight`` after it turns on.
            motif_scales: Positive pooling scales used by the motif descriptor.
            patch_weight: Peak multiplier for local patch-vocabulary matching.
            patch_weight_end: Patch multiplier at the finest stage. ``None``
                keeps ``patch_weight`` after it turns on.
            patch_sizes: Odd local patch widths used by vocabulary matching.
            patch_stride: Spatial stride used when extracting vocabulary patches.

        Returns:
            This model, for chaining.
        """
        occ = torch.as_tensor(occupancy, dtype=torch.float32)
        if occ.ndim == 3 and occ.shape[0] == 1:
            occ = occ[0]
        if occ.ndim != 2:
            raise ValueError(
                f'sketch occupancy must be 2-D, got shape {tuple(occ.shape)}')
        self.sketch_occupancy_full = occ.clamp(0.0, 1.0).cpu().contiguous()
        self.sketch_weight_start = float(weight)
        self.sketch_weight_end = (
            None if weight_end is None else float(weight_end))
        self.sketch_weight = float(weight)
        self.sketch_motif_weight_peak = float(motif_weight)
        self.sketch_motif_weight_end = (
            None if motif_weight_end is None else float(motif_weight_end))
        self.sketch_motif_weight = (
            float(motif_weight) if not hasattr(self, 'resize_num') else 0.0)
        self.sketch_motif_scales = tuple(int(scale) for scale in motif_scales)
        self.sketch_patch_weight_peak = float(patch_weight)
        self.sketch_patch_weight_end = (
            None if patch_weight_end is None else float(patch_weight_end))
        self.sketch_patch_weight = (
            float(patch_weight) if not hasattr(self, 'resize_num') else 0.0)
        self.sketch_patch_sizes = tuple(int(size) for size in patch_sizes)
        self.sketch_patch_stride = int(patch_stride)
        return self

    def sketch_weight_at(
        self,
        *,
        step: int = 0,
        max_iterations: int = 1,
    ) -> float:
        """Sketch multiplier for the current AdaptivePixel stage, or step.

        AdaptivePixel: linear in ``resizes / resize_num`` so the coarse
        grid gets ``sketch_weight_start`` and the finest gets
        ``sketch_weight_end``. PixelModel (no schedule): linear in
        ``step / (max_iterations - 1)``. ``weight_end is None`` is constant.
        """
        start = float(self.sketch_weight_start)
        end = self.sketch_weight_end
        if end is None:
            return start
        end = float(end)
        resize_num = getattr(self, 'resize_num', None)
        resizes = getattr(self, 'resizes', None)
        if resize_num is not None and int(resize_num) > 0 and resizes is not None:
            t = float(resizes) / float(resize_num)
            return (1.0 - t) * start + t * end
        denom = max(int(max_iterations) - 1, 1)
        t = float(step) / float(denom)
        t = min(max(t, 0.0), 1.0)
        return (1.0 - t) * start + t * end

    def apply_sketch_schedule(
        self,
        *,
        step: int = 0,
        max_iterations: int = 1,
    ) -> float:
        """Set spatial and motif sketch weights for this optimization step."""
        if self.sketch_occupancy_full is None:
            return float(self.sketch_weight)
        weight = self.sketch_weight_at(
            step=step, max_iterations=max_iterations)
        self.sketch_weight = weight
        peak = float(self.sketch_motif_weight_peak)
        motif_end = (
            peak if self.sketch_motif_weight_end is None
            else float(self.sketch_motif_weight_end))
        resize_num = getattr(self, 'resize_num', None)
        resizes = getattr(self, 'resizes', None)
        if resize_num is not None and int(resize_num) > 0 and resizes is not None:
            # Establish global shape without a texture-style term. Turn the
            # motif on after the first upsample, then ease it toward the final
            # value so compliance can clean up local members.
            if int(resizes) == 0:
                self.sketch_motif_weight = 0.0
            elif int(resizes) >= int(resize_num):
                self.sketch_motif_weight = motif_end
            else:
                progress = (int(resizes) - 1) / max(int(resize_num) - 1, 1)
                self.sketch_motif_weight = (
                    (1.0 - progress) * peak + progress * motif_end)
        else:
            self.sketch_motif_weight = peak
        patch_peak = float(self.sketch_patch_weight_peak)
        patch_end = (
            patch_peak if self.sketch_patch_weight_end is None
            else float(self.sketch_patch_weight_end))
        if resize_num is not None and int(resize_num) > 0 and resizes is not None:
            if int(resizes) == 0:
                self.sketch_patch_weight = 0.0
            elif int(resizes) >= int(resize_num):
                self.sketch_patch_weight = patch_end
            else:
                progress = (int(resizes) - 1) / max(int(resize_num) - 1, 1)
                self.sketch_patch_weight = (
                    (1.0 - progress) * patch_peak + progress * patch_end)
        else:
            self.sketch_patch_weight = patch_peak
        return weight

    def _sketch_prior_active(self) -> bool:
        return (
            self.sketch_occupancy_full is not None
            and float(self.sketch_weight) != 0.0
        )

    def _sketch_motif_active(self) -> bool:
        return (
            self.sketch_occupancy_full is not None
            and float(self.sketch_motif_weight) != 0.0
        )

    def _sketch_patch_active(self) -> bool:
        return (
            self.sketch_occupancy_full is not None
            and float(self.sketch_patch_weight) != 0.0
        )

    def _sketch_guidance_active(self) -> bool:
        return (
            self._sketch_prior_active()
            or self._sketch_motif_active()
            or self._sketch_patch_active()
        )

    def _occupancy_on_density(self, density: torch.Tensor) -> torch.Tensor:
        """Resample stored occupancy onto ``density``'s spatial grid."""
        occ = self.sketch_occupancy_full.to(
            device=density.device, dtype=density.dtype)
        height, width = int(density.shape[-2]), int(density.shape[-1])
        if occ.shape[-2] != height or occ.shape[-1] != width:
            occ = F.interpolate(
                occ.view(1, 1, occ.shape[-2], occ.shape[-1]),
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            ).view(height, width)
        while occ.ndim < density.ndim:
            occ = occ.unsqueeze(0)
        return occ.expand_as(density)

    def _load_sites_on_density(self, density: torch.Tensor) -> torch.Tensor:
        """One element per loaded node, resampled onto ``density``'s grid."""
        nely = int(self.env.args['nely'])
        nelx = int(self.env.args['nelx'])
        mask_np = load_site_mask(
            self.env.args['forces'], nely=nely, nelx=nelx)
        sites = torch.as_tensor(
            mask_np, device=density.device, dtype=density.dtype)
        height, width = int(density.shape[-2]), int(density.shape[-1])
        if sites.shape[-2] != height or sites.shape[-1] != width:
            sites = F.interpolate(
                sites.view(1, 1, sites.shape[-2], sites.shape[-1]),
                size=(height, width),
                mode='nearest',
            ).view(height, width)
        while sites.ndim < density.ndim:
            sites = sites.unsqueeze(0)
        return sites.expand_as(density)

    def get_sketch_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Unweighted mass-prior loss on the canonical physical density.

        Template is occupancy ∪ load-application pixels. Returns a zero
        tensor (no ``PhysicalDensity`` eval) when the prior is off, so a
        Venice-preset run without a sketch stays bit-identical.
        """
        if not self._sketch_prior_active():
            return logits.new_tensor(0.0)
        density = self.get_physical_density(logits)
        occupancy = self._occupancy_on_density(density)
        load_sites = self._load_sites_on_density(density)
        return sketch_mass_prior_loss(
            density, occupancy, load_sites=load_sites)

    def get_sketch_motif_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Unweighted, translation-invariant local motif loss."""
        if self.sketch_occupancy_full is None:
            return logits.new_tensor(0.0)
        density = self.get_physical_density(logits)
        occupancy = self._occupancy_on_density(density)
        return sketch_motif_loss(
            density, occupancy, scales=self.sketch_motif_scales)

    def get_sketch_patch_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Unweighted, translation-invariant patch-vocabulary loss."""
        if self.sketch_occupancy_full is None:
            return logits.new_tensor(0.0)
        density = self.get_physical_density(logits)
        occupancy = self._occupancy_on_density(density)
        load_sites = self._load_sites_on_density(density)
        return sketch_patch_vocabulary_loss(
            density,
            occupancy,
            load_sites=load_sites,
            patch_sizes=self.sketch_patch_sizes,
            stride=self.sketch_patch_stride,
        )

    def add_sketch_term(
        self, loss: torch.Tensor, logits: torch.Tensor,
    ) -> torch.Tensor:
        """Add ``sketch_weight * get_sketch_loss`` when the prior is on.

        Identity when the prior is off. Optimizer paths that compose the
        total by hand (Adam/LBFGS ``clip_alpha``, AdaptiveAdam's default
        ``_compose_loss``) must call this; paths that already go through
        ``get_total_loss`` / ``get_venice_compat_losses`` must not, or the
        term is applied twice.
        """
        if not self._sketch_guidance_active():
            return loss
        # Normal composition computes the physical density once even when
        # spatial and motif guidance are both active.
        density = self.get_physical_density(logits)
        occupancy = self._occupancy_on_density(density)
        total = loss
        load_sites = None
        if self._sketch_prior_active() or self._sketch_patch_active():
            load_sites = self._load_sites_on_density(density)
        if self._sketch_prior_active():
            total = total + float(self.sketch_weight) * sketch_mass_prior_loss(
                density, occupancy, load_sites=load_sites)
        if self._sketch_motif_active():
            total = total + float(self.sketch_motif_weight) * sketch_motif_loss(
                density, occupancy, scales=self.sketch_motif_scales)
        if self._sketch_patch_active():
            total = (
                total
                + float(self.sketch_patch_weight) * sketch_patch_vocabulary_loss(
                    density,
                    occupancy,
                    load_sites=load_sites,
                    patch_sizes=self.sketch_patch_sizes,
                    stride=self.sketch_patch_stride,
                )
            )
        return total

    def enable_venice_compat_loss(
        self,
        algebra: VeniceLossAlgebra | bool = True,
    ) -> "Model":
        """Switch `get_total_loss` to the legacy Venice algebra.

        Provided because the model subclasses do not forward the `venice_compat`
        constructor kwarg, so configuration wiring can flip the seam on an
        already-built model.

        Args:
            algebra: True for the reference coefficients, an explicit
                :class:`VeniceLossAlgebra`, or False to restore the default.

        Returns:
            This model, for chaining.
        """
        self.venice_loss_algebra = _coerce_venice_loss_algebra(algebra)
        return self

    def get_venice_compat_losses(
        self,
        logits: torch.Tensor,
        clip_alpha: Optional[float] = None,
        compliance_weight: Optional[float] = None,
    ) -> VeniceLossTerms:
        """Compute the full legacy loss breakdown for one step.

        Exposed separately from `get_total_loss` so a run can log the same
        fields as the reference trajectory (compliance, weighted CLIP, raw CLIP
        and the total). Usable whether or not the seam is enabled; arguments
        left as None fall back to this model's algebra, then to
        :data:`VENICE_LOSS_ALGEBRA`.
        """
        algebra = self.venice_loss_algebra or VENICE_LOSS_ALGEBRA
        terms = venice_compat_total_loss(
            self.get_structural_loss(logits),
            self.get_semantic_loss(logits),
            clip_alpha=algebra.clip_alpha if clip_alpha is None else clip_alpha,
            compliance_weight=(
                algebra.compliance_weight if compliance_weight is None else compliance_weight
            ),
        )
        return terms._replace(
            total_loss=self.add_sketch_term(terms.total_loss, logits))

    def get_total_loss(
        self, 
        logits: torch.Tensor, 
        clip_weight: Optional[float] = None,
        dynamic_clip_alpha: Optional[float] = None,
        compliance_weight: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute combined loss with an optional semantic weight.
        
        Args:
            logits: Raw design logits.
            clip_weight: Scalar multiplier for semantic loss. If None, uses
                `self.clip_weight_default` (defaults to 1.0). Ignored when
                `clip_loss` is absent.
            dynamic_clip_alpha: If provided, scales the semantic CLIP loss weight
                proportionally to the current structural loss via
                `effective_weight = dynamic_clip_alpha * structural_loss.detach()`.
                Overrides `clip_weight` when not None.
            compliance_weight: Optional scalar to scale the structural loss term,
                and to determine the dynamic CLIP weight when `dynamic_clip_alpha`
                is provided (i.e., weight is based on the scaled structural loss).

        When the Venice compatibility seam is enabled, the legacy algebra
        replaces all of the above: `dynamic_clip_alpha` and `compliance_weight`
        override the configured algebra when given, the weight is undetached,
        and `clip_weight` is REFUSED rather than ignored (see below). See
        :func:`venice_compat_total_loss`.

        Raises:
            ValueError: if `clip_weight` is given while the seam is enabled.
                The legacy weight is `compliance * clip_alpha`, recomputed
                every step; a static `clip_weight` is not a setting of that
                formula but a different one, so a caller passing it is
                describing a different run and gets an error rather than
                whichever weight the branch happens to apply. This mirrors the
                refusal `train.optimizers` already applies to `clip_alpha`.
        """
        if self.venice_loss_algebra is not None:
            if clip_weight is not None:
                raise ValueError(
                    f'get_total_loss got clip_weight={clip_weight!r} while the '
                    'Venice compatibility algebra is enabled; those are '
                    'contradictory couplings. The algebra weights the semantic '
                    'loss by compliance * clip_alpha, recomputed and '
                    'undetached at every step, so a static clip_weight has no '
                    'place in it and would be silently discarded. Drop '
                    'clip_weight to run the preset -- set its alpha with '
                    'enable_venice_compat_loss(VeniceLossAlgebra(clip_alpha=...)) '
                    '-- or disable the preset to use the default coupling.')
            return self.get_venice_compat_losses(
                logits,
                clip_alpha=dynamic_clip_alpha,
                compliance_weight=compliance_weight,
            ).total_loss

        structural_loss = self.get_structural_loss(logits)
        structural_loss_eff = structural_loss if compliance_weight is None else (structural_loss * float(compliance_weight))
        semantic_loss = self.get_semantic_loss(logits)
        if self.clip_loss is None:
            return self.add_sketch_term(structural_loss_eff, logits)
        base_w = self.clip_weight_default if clip_weight is None else float(clip_weight)
        if dynamic_clip_alpha is not None:
            # Couple CLIP guidance to current (optionally scaled) compliance
            w_eff = float(dynamic_clip_alpha) * structural_loss_eff.detach()
        else:
            w_eff = base_w
        return self.add_sketch_term(
            structural_loss_eff + semantic_loss * w_eff, logits)
