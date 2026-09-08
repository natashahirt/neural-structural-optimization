"""Live semantic spatial prior: CLIP/SDS saliency as an occupancy map.

CLIP (or a frozen diffusion score) proposes *where* material should sit.
Exact FEA compliance is unchanged as a scalar; a Stage-6-style mass-prior
term attracts volume toward the evolving preference map. Preference is
``relu(-dL/d rho)`` on canonical physical density -- never inferred from
displayed black/white polarity.

Scales can optionally be scored on a filter-then-project view of the density
so the motif cannot be satisfied by mechanically negligible gray. Physics
always sees the unprojected field.

Disabled / zero-weight is a no-op: no extra CLIP/SDS forwards, no extra
``PhysicalDensity`` eval, occupancy left unset.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_structural_optimization.experiment import physical_motif_scale_fracs
from neural_structural_optimization.models.loss_sketch import sketch_mass_prior_loss

SCALE_NAMES = ('global', 'storey', 'member')

# DDPM-style noise bands mapped onto the physical hierarchy. Higher noise
# is a coarser semantic proposal.
SDS_TIMESTEP_BANDS = {
    'global': (700, 900),
    'storey': (300, 500),
    'member': (50, 200),
}
SDS_NUM_TIMESTEPS = 1000

# Cells below this physical density carry little load, so preference mass
# sitting there is semantic "ink" rather than structure.
INK_DENSITY_THRESHOLD = 0.3


def scale_fracs_for_grid(
    height: int,
    interval: int,
) -> dict[str, float]:
    """Building / storey / member elevation fractions for this grid."""
    building, storey, member = physical_motif_scale_fracs(height, interval)
    return {'global': float(building), 'storey': float(storey), 'member': float(member)}


def gaussian_blur2d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable isotropic Gaussian. ``sigma<=0`` is identity."""
    if float(sigma) <= 0.0:
        return image
    if image.ndim == 2:
        x = image.unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif image.ndim == 3:
        x = image.unsqueeze(0)
        squeezed = True
    else:
        x = image
        squeezed = False
    k = max(3, int(2 * round(3.0 * float(sigma)) + 1))
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2.0
    kern = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kern = kern / kern.sum().clamp_min(1e-12)
    ky = kern.view(1, 1, k, 1)
    kx = kern.view(1, 1, 1, k)
    pad = k // 2
    channels = x.shape[1]
    x = F.conv2d(x, ky.expand(channels, 1, k, 1), padding=(pad, 0), groups=channels)
    x = F.conv2d(x, kx.expand(channels, 1, 1, k), padding=(0, pad), groups=channels)
    if squeezed:
        return x.view(image.shape)
    return x


def heaviside_projection(
    density: torch.Tensor,
    beta: float,
    eta: float = 0.5,
) -> torch.Tensor:
    """Smooth Heaviside projection about ``eta`` (Wang/Lazarov/Sigmund tanh form).

    Stays differentiable, but ``d proj / d rho`` peaks at ``eta`` and decays
    sharply in the faint tail. Scoring semantics on this view therefore pays
    almost nothing for near-void density. ``beta <= 0`` is identity.
    """
    beta = float(beta)
    if beta <= 0.0:
        return density
    eta = float(eta)
    offset = float(np.tanh(beta * eta))
    denom = offset + float(np.tanh(beta * (1.0 - eta)))
    return (offset + torch.tanh(beta * (density - eta))) / denom


def projected_density_view(
    density: torch.Tensor,
    *,
    beta: float,
    eta: float = 0.5,
    filter_sigma: float = 0.0,
) -> torch.Tensor:
    """Filter-then-project view of density, for semantic scoring only.

    The blur imposes a minimum feature size and the projection removes the
    faint-gray tail, so the only way to improve the semantic score is to
    commit material wider than ``filter_sigma`` at close to full density.
    Physics still sees the unprojected field.
    """
    return heaviside_projection(
        gaussian_blur2d(density, filter_sigma), beta, eta)


def sigma_for_min_feature(width: int, min_feature_frac: float) -> float:
    """Blur sigma erasing features narrower than ``min_feature_frac * width``.

    Expressed relative to the domain so the minimum feature stays physically
    constant as the hierarchical curriculum upsamples the grid. A feature of
    width ``w`` is suppressed by a Gaussian of roughly ``w / 2``.
    """
    if float(min_feature_frac) <= 0.0:
        return 0.0
    return max(0.5, 0.5 * float(min_feature_frac) * float(width))


def preference_from_score(
    density: torch.Tensor,
    score: torch.Tensor,
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    """``relu(-dL/d rho)``: cells where adding material lowers the score.

    ``density`` must require grad and be an ancestor of ``score``. The
    returned map is detached.
    """
    if score.ndim != 0:
        score = score.reshape(()).sum()
    grad, = torch.autograd.grad(
        score, density, retain_graph=retain_graph, create_graph=False)
    return torch.relu(-grad).detach()


def normalize_preference(preference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map a non-negative preference field into ``[0, 1]`` by its peak."""
    peak = preference.amax().clamp_min(eps)
    return (preference / peak).clamp(0.0, 1.0)


def cosine_alignment(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine of two tensors flattened to a vector."""
    va = a.reshape(-1).float()
    vb = b.reshape(-1).float()
    denom = va.norm().clamp_min(eps) * vb.norm().clamp_min(eps)
    return (va * vb).sum() / denom


def gradient_norm_ratio(guidance: torch.Tensor, compliance: torch.Tensor) -> float:
    """``||g_guide|| / ||g_compliance||`` for weight calibration."""
    cg = float(compliance.reshape(-1).float().norm().clamp_min(1e-12))
    return float(guidance.reshape(-1).float().norm() / cg)


def connectivity_metrics(
    density: np.ndarray,
    load_sites: np.ndarray,
    *,
    threshold: float = 0.3,
) -> dict[str, float]:
    """Support-to-load connectivity and floating-component mass.

    Bottom row of the elevation is the ground support of
    ``multistory_building``. A component is load-bearing if it touches both
    that row and a load-application pixel.
    """
    from scipy import ndimage

    dens = np.asarray(density, dtype=np.float64)
    while dens.ndim > 2:
        dens = dens[0]
    sites = np.asarray(load_sites, dtype=bool)
    while sites.ndim > 2:
        sites = sites[0]
    binary = dens >= float(threshold)
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return {
            'component_count': 0.0,
            'top_to_bottom_connected': 0.0,
            'support_to_load_connected': 0.0,
            'floating_mass_fraction': float(dens.sum() > 0.0),
            'spanning_mass_fraction': 0.0,
        }
    bottom = set(labels[-1][labels[-1] > 0].tolist())
    top = set(labels[0][labels[0] > 0].tolist())
    load_labels = set(labels[sites].tolist()) if sites.any() else set()
    load_labels.discard(0)
    spanning = bottom & top
    load_bearing = bottom & load_labels
    mass = dens.sum()
    spanning_mass = dens[np.isin(labels, list(spanning))].sum() if spanning else 0.0
    grounded = dens[np.isin(labels, list(bottom))].sum() if bottom else 0.0
    floating = max(mass - grounded, 0.0)
    return {
        'component_count': float(count),
        'top_to_bottom_connected': float(bool(spanning)),
        'support_to_load_connected': float(bool(load_bearing)),
        'floating_mass_fraction': float(floating / max(mass, 1e-12)),
        'spanning_mass_fraction': float(spanning_mass / max(mass, 1e-12)),
    }


class SemanticScoreProvider:
    """Scalar semantic loss on a physical-density field, optionally per scale."""

    def score_by_scale(
        self,
        density: torch.Tensor,
        scales: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class CallableScoreProvider(SemanticScoreProvider):
    """Test double: a callable (and optional per-scale callables) on density."""

    def __init__(
        self,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        scale_fns: Optional[Mapping[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    ):
        self.score_fn = score_fn
        self.scale_fns = dict(scale_fns or {})

    def score_by_scale(self, density, scales):
        out = {}
        for name in scales:
            fn = self.scale_fns.get(name, self.score_fn)
            out[str(name)] = fn(density)
        return out


class CLIPSemanticProvider(SemanticScoreProvider):
    """Wrap a ``CLIPLoss`` so each physical scale has its own scalar."""

    def __init__(self, clip_loss, scale_fracs: Mapping[str, float]):
        self.clip_loss = clip_loss
        self.scale_fracs = {str(k): float(v) for k, v in scale_fracs.items()}

    def _global_score(self, density: torch.Tensor) -> torch.Tensor:
        saved = tuple(self.clip_loss.motif_scale_fracs)
        self.clip_loss.motif_scale_fracs = ()
        try:
            return self.clip_loss(density)
        finally:
            self.clip_loss.motif_scale_fracs = saved

    def score_by_scale(self, density, scales):
        out = {}
        for name in scales:
            if name == 'global':
                out[name] = self._global_score(density)
                continue
            frac = self.scale_fracs[name]
            saved = tuple(self.clip_loss.motif_scale_fracs)
            saved_losses = dict(getattr(self.clip_loss, 'last_motif_scale_losses', {}))
            self.clip_loss.motif_scale_fracs = (frac,)
            try:
                out[name] = self.clip_loss._physical_motif_scale_loss(density)
            finally:
                self.clip_loss.motif_scale_fracs = saved
                self.clip_loss.last_motif_scale_losses = saved_losses
        return out


def _sinusoidal_timestep(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
    args = t.float().reshape(-1, 1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FrozenDenoiser(nn.Module):
    """Tiny frozen conv denoiser for SDS. Not a pretrained image prior.

    Used so the SDS *interface* (noise bands, residual, occupancy EMA) can
    be tested and run without downloading Stable Diffusion. A real UNet can
    replace it via ``DiffusionSDSProvider(denoiser=...)``.
    """

    def __init__(self, channels: int = 16, cond_dim: int = 0, seed: int = 0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.time_dim = channels
        in_ch = 1 + (1 if cond_dim else 0)
        self.in_conv = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.mid = nn.Conv2d(channels, channels, 3, padding=1)
        self.out_conv = nn.Conv2d(channels, 1, 3, padding=1)
        self.time_proj = nn.Linear(self.time_dim, channels)
        if cond_dim:
            self.cond_proj = nn.Linear(cond_dim, 1)
        else:
            self.cond_proj = None
        cpu_gen = torch.Generator(device='cpu')
        cpu_gen.manual_seed(int(seed))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data.copy_(
                    torch.randn(module.weight.shape, generator=cpu_gen) * 0.05)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.requires_grad_(False)
        self.eval()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
        if x_t.shape[1] != 1:
            x_t = x_t.mean(dim=1, keepdim=True)
        if self.cond_proj is not None:
            if cond is None:
                cond_map = x_t.new_zeros(1, 1, x_t.shape[-2], x_t.shape[-1])
            else:
                cond_map = self.cond_proj(cond.reshape(1, -1).to(x_t.dtype)).view(1, 1, 1, 1)
                cond_map = cond_map.expand(-1, 1, x_t.shape[-2], x_t.shape[-1])
            x_in = torch.cat([x_t, cond_map], dim=1)
        else:
            x_in = x_t
        h = F.silu(self.in_conv(x_in))
        t_emb = self.time_proj(_sinusoidal_timestep(t.to(x_t.device), self.time_dim).to(x_t.dtype))
        h = h + t_emb.view(-1, h.shape[1], 1, 1)
        h = F.silu(self.mid(h))
        return self.out_conv(h)


class DiffusionSDSProvider(SemanticScoreProvider):
    """Score-distillation sampling on physical density, banded by scale.

    ``score`` is the SDS surrogate ``((eps_pred - eps).detach() * x_t).mean()``
    so ``autograd.grad`` w.r.t. the clean density recovers the usual SDS
    direction without differentiating the frozen denoiser.
    """

    def __init__(
        self,
        denoiser: Optional[nn.Module] = None,
        *,
        cond: Optional[torch.Tensor] = None,
        timestep_bands: Optional[Mapping[str, tuple[int, int]]] = None,
        num_timesteps: int = SDS_NUM_TIMESTEPS,
        seed: int = 0,
    ):
        self.denoiser = denoiser if denoiser is not None else FrozenDenoiser(seed=seed)
        self.denoiser.eval().requires_grad_(False)
        self.cond = cond
        self.timestep_bands = dict(timestep_bands or SDS_TIMESTEP_BANDS)
        self.num_timesteps = int(num_timesteps)
        self.seed = int(seed)
        self._step = 0
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        self.register_buffers = False
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def _sample_t(self, scale: str, device) -> torch.Tensor:
        lo, hi = self.timestep_bands[scale]
        lo = max(1, int(lo))
        hi = min(int(hi), self.num_timesteps - 1)
        scale_id = {'global': 1, 'storey': 2, 'member': 3}.get(scale, 0)
        t = lo + (self._step + scale_id) % max(hi - lo, 1)
        self._step += 1
        return torch.tensor([t], device=device, dtype=torch.long)

    def _sds_score(self, density: torch.Tensor, scale: str) -> torch.Tensor:
        x = density
        while x.ndim < 4:
            x = x.unsqueeze(0)
        x = x.clamp(0.0, 1.0)
        t = self._sample_t(scale, x.device)
        alpha_bar = self.alpha_bar.to(device=x.device, dtype=x.dtype)[t.clamp(0, self.num_timesteps - 1)]
        sqrt_ab = alpha_bar.sqrt().view(-1, 1, 1, 1)
        sqrt_om = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed + self._step)
        eps = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(device=x.device, dtype=x.dtype)
        x_t = sqrt_ab * x + sqrt_om * eps
        with torch.no_grad():
            eps_pred = self.denoiser(x_t, t, self.cond)
        residual = (eps_pred - eps).detach()
        if residual.shape != x_t.shape:
            residual = residual.expand_as(x_t)
        return (residual * x_t).mean()

    def score_by_scale(self, density, scales):
        return {str(name): self._sds_score(density, str(name)) for name in scales}


class SemanticSpatialPrior:
    """EMA occupancy maps, one per physical scale, blended for the mass prior.

    Scales named in ``projection_scales`` are scored on a filter-then-project
    view of the density instead of the raw field. That is the sculptural
    setting: the motif can only score by committing material at member width,
    not by laying down faint gray. Scales left out keep the raw view, so the
    later detail stages can still resolve fine texture. ``projection_beta=0``
    disables projection everywhere and restores the unprojected behaviour.
    """

    def __init__(
        self,
        provider: SemanticScoreProvider,
        *,
        scale_fracs: Mapping[str, float],
        weight: float = 0.0,
        auto_ratio: Optional[float] = None,
        ema_decay: float = 0.9,
        smooth_sigma: float = 1.5,
        curriculum: str = 'global_only',
        record_alignment: bool = False,
        projection_beta: float = 0.0,
        projection_eta: float = 0.5,
        projection_filter_sigma: float = 0.0,
        projection_scales: Sequence[str] = ('global',),
    ):
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError(f'ema_decay must be in [0, 1), got {ema_decay}')
        if curriculum not in ('global_only', 'all', 'hierarchical'):
            raise ValueError(
                f"curriculum must be 'global_only', 'all', or 'hierarchical', "
                f'got {curriculum!r}')
        if not 0.0 < float(projection_eta) < 1.0:
            raise ValueError(
                f'projection_eta must be in (0, 1), got {projection_eta}')
        projection_scales = tuple(str(name) for name in projection_scales)
        unknown = set(projection_scales) - set(SCALE_NAMES)
        if unknown:
            raise ValueError(
                f'unknown projection_scales {sorted(unknown)}, '
                f'expected a subset of {SCALE_NAMES}')
        self.provider = provider
        self.scale_fracs = {str(k): float(v) for k, v in scale_fracs.items()}
        self.weight = float(weight)
        self.auto_ratio = None if auto_ratio is None else float(auto_ratio)
        self.ema_decay = float(ema_decay)
        self.smooth_sigma = float(smooth_sigma)
        self.curriculum = str(curriculum)
        self.record_alignment = bool(record_alignment)
        self.projection_beta = float(projection_beta)
        self.projection_eta = float(projection_eta)
        self.projection_filter_sigma = float(projection_filter_sigma)
        self.projection_scales = projection_scales
        self.maps: dict[str, Optional[torch.Tensor]] = {name: None for name in SCALE_NAMES}
        self.last_instant: dict[str, torch.Tensor] = {}
        self.last_metrics: dict[str, float] = {}
        self.last_clip_grad: Optional[torch.Tensor] = None
        self._calibrated = False

    def is_active(self) -> bool:
        if self.auto_ratio is not None and not self._calibrated:
            return True
        return float(self.weight) != 0.0

    def projects(self, scale: str) -> bool:
        """Whether ``scale`` is scored on the filter-then-project view."""
        return self.projection_beta > 0.0 and str(scale) in self.projection_scales

    def _score_by_scale(
        self,
        density: torch.Tensor,
        scales: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Score each scale on its own view, projected or raw, of ``density``."""
        projected = tuple(name for name in scales if self.projects(name))
        raw = tuple(name for name in scales if name not in projected)
        scores: dict[str, torch.Tensor] = {}
        if raw:
            scores.update(self.provider.score_by_scale(density, raw))
        if projected:
            view = projected_density_view(
                density,
                beta=self.projection_beta,
                eta=self.projection_eta,
                filter_sigma=self.projection_filter_sigma,
            )
            scores.update(self.provider.score_by_scale(view, projected))
        return scores

    def active_scales(
        self,
        *,
        resizes: Optional[int] = None,
        resize_num: Optional[int] = None,
        step: int = 0,
        max_iterations: int = 1,
    ) -> tuple[str, ...]:
        names = tuple(name for name in SCALE_NAMES if name in self.scale_fracs)
        if self.curriculum == 'global_only':
            return ('global',) if 'global' in self.scale_fracs else names[:1]
        if self.curriculum == 'all':
            return names
        if resize_num is not None and resizes is not None and int(resize_num) > 0:
            if int(resizes) <= 0:
                return ('global',)
            if int(resizes) == 1:
                return tuple(n for n in ('global', 'storey') if n in self.scale_fracs)
            return names
        denom = max(int(max_iterations) - 1, 1)
        t = float(step) / float(denom)
        if t < 1.0 / 3.0:
            return ('global',)
        if t < 2.0 / 3.0:
            return tuple(n for n in ('global', 'storey') if n in self.scale_fracs)
        return names

    def _resize_map(self, field: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if field.shape[-2] == height and field.shape[-1] == width:
            return field
        x = field
        while x.ndim < 4:
            x = x.unsqueeze(0)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        while x.ndim > field.ndim:
            x = x.squeeze(0)
        return x

    def update(
        self,
        density: torch.Tensor,
        *,
        resizes: Optional[int] = None,
        resize_num: Optional[int] = None,
        step: int = 0,
        max_iterations: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Refresh EMA maps from ``relu(-dL/d rho)``. Occupancy is detached."""
        field = density
        if field.ndim == 3 and field.shape[0] == 1:
            field_2d = field[0]
        elif field.ndim == 2:
            field_2d = field
        else:
            field_2d = field.reshape(field.shape[-2], field.shape[-1])
        work = field_2d.detach().requires_grad_(True)
        scales = self.active_scales(
            resizes=resizes, resize_num=resize_num,
            step=step, max_iterations=max_iterations)
        scores = self._score_by_scale(work, scales)
        height, width = int(work.shape[-2]), int(work.shape[-1])
        combined = None
        instant = {}
        for i, name in enumerate(scales):
            pref = preference_from_score(
                work, scores[name], retain_graph=i < len(scales) - 1)
            pref = gaussian_blur2d(normalize_preference(pref), self.smooth_sigma)
            instant[name] = pref
            prev = self.maps[name]
            if prev is None:
                ema = pref
            else:
                prev = self._resize_map(prev.to(device=pref.device, dtype=pref.dtype), height, width)
                ema = self.ema_decay * prev + (1.0 - self.ema_decay) * pref
            self.maps[name] = ema.detach()
            combined = pref if combined is None else combined + pref
        self.last_instant = instant
        if combined is None:
            combined = torch.zeros_like(work)
        # Stacked CLIP preference used for overlap / polarity diagnostics.
        self.last_clip_grad = combined.detach()
        occupancy = normalize_preference(sum(
            (self.maps[name] for name in scales if self.maps[name] is not None),
            torch.zeros_like(work),
        ))
        delta = []
        for name in scales:
            prev = self.maps[name]
            inst = instant.get(name)
            if prev is not None and inst is not None:
                delta.append(float((prev - inst).abs().mean()))
        self.last_metrics = {
            'active_scale_count': float(len(scales)),
            'occupancy_mean': float(occupancy.mean()),
            'ema_delta': float(np.mean(delta) if delta else 0.0),
            'weight': float(self.weight),
            'projection_beta': float(self.projection_beta),
            'projected_scale_count': float(
                sum(1 for name in scales if self.projects(name))),
            'preference_ink_fraction': self._ink_fraction(work, combined),
        }
        for name in scales:
            self.last_metrics[f'occupancy_mean_{name}'] = float(self.maps[name].mean())
        return {name: self.maps[name] for name in scales}

    @staticmethod
    def _ink_fraction(
        density: torch.Tensor,
        preference: torch.Tensor,
        threshold: float = INK_DENSITY_THRESHOLD,
    ) -> float:
        """Share of preference mass asking for material in near-void cells.

        High values mean the motif is being satisfied by faint gray that
        carries no load -- the failure mode projection is meant to close.
        """
        total = preference.sum()
        if float(total) <= 0.0:
            return 0.0
        faint = (density.detach() < float(threshold)).to(preference.dtype)
        return float((preference * faint).sum() / total)

    def blended_occupancy(self, density: torch.Tensor) -> torch.Tensor:
        """Detached ``[0, 1]`` occupancy on ``density``'s grid."""
        height, width = int(density.shape[-2]), int(density.shape[-1])
        device = density.device
        dtype = density.dtype
        acc = None
        for name in SCALE_NAMES:
            field = self.maps[name]
            if field is None:
                continue
            field = self._resize_map(field.to(device=device, dtype=dtype), height, width)
            acc = field if acc is None else acc + field
        if acc is None:
            acc = torch.zeros((height, width), device=device, dtype=dtype)
        occ = normalize_preference(acc)
        while occ.ndim < density.ndim:
            occ = occ.unsqueeze(0)
        return occ.expand_as(density).detach()

    def mass_prior_loss(
        self,
        density: torch.Tensor,
        load_sites: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        occupancy = self.blended_occupancy(density)
        return sketch_mass_prior_loss(density, occupancy, load_sites=load_sites)

    def maybe_calibrate_weight(
        self,
        guidance_grad: torch.Tensor,
        compliance_grad: torch.Tensor,
    ) -> float:
        """Set ``weight`` so ``||w g_prior|| / ||g_C||`` matches ``auto_ratio``."""
        if self.auto_ratio is None or self._calibrated:
            return float(self.weight)
        ratio = gradient_norm_ratio(guidance_grad, compliance_grad)
        if ratio <= 0.0:
            self.weight = 0.0
        else:
            self.weight = float(self.auto_ratio) / ratio
        self._calibrated = True
        self.last_metrics['weight'] = float(self.weight)
        self.last_metrics['grad_ratio_uncalibrated'] = float(ratio)
        return float(self.weight)
