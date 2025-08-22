from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import clip
import math
from typing import Sequence, Tuple

# ---- multi-patch pyramid and tiling helpers ----
def _affine_grid_from_boxes(H: int, W: int, boxes_xywh: torch.Tensor, out: Tuple[int,int]):
    """
    boxes_xywh: [N,4] in absolute pixels (cx, cy, w, h) on the input image (H,W).
    Returns sampling grid for F.grid_sample to produce [N, C, outH, outW].
    """
    assert boxes_xywh.ndim == 2 and boxes_xywh.shape[1] == 4
    cx = boxes_xywh[:, 0] * (2.0 / W) - 1.0
    cy = boxes_xywh[:, 1] * (2.0 / H) - 1.0
    sx = boxes_xywh[:, 2] / W
    sy = boxes_xywh[:, 3] / H
    # Map output ([-1,1]) -> selected window centered at (cx,cy) with half-width sx and half-height sy
    # Note: for affine_grid, the scale maps full [-1,1] span to 2*sx in normalized coords, so use sx, sy directly.
    theta = boxes_xywh.new_zeros((boxes_xywh.size(0), 2, 3))
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    theta[:, 0, 2] = cx
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, size=(boxes_xywh.size(0), 3, out[0], out[1]), align_corners=False)
    return grid

def _fixed_grid_boxes(H: int, W: int, patch_px: int, stride_px: int, jitter_frac: float = 0.0, device=None, dtype=None):
    """
    Create a fixed, overlapping grid of (cx, cy, w, h) boxes in pixels covering the image.
    patch_px: square crop (w=h=patch_px at input scale)
    stride_px: stride between patch centers
    jitter_frac: random jitter up to +/- jitter_frac * stride_px on centers (per patch)
    """
    xs = torch.arange(patch_px//2, W - patch_px//2 + 1, stride_px, device=device, dtype=dtype)
    ys = torch.arange(patch_px//2, H - patch_px//2 + 1, stride_px, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    cx = gx.reshape(-1)
    cy = gy.reshape(-1)
    if jitter_frac > 0:
        jx = (torch.rand_like(cx) - 0.5) * 2.0 * (jitter_frac * stride_px)
        jy = (torch.rand_like(cy) - 0.5) * 2.0 * (jitter_frac * stride_px)
        cx = (cx + jx).clamp(patch_px/2, W - patch_px/2)
        cy = (cy + jy).clamp(patch_px/2, H - patch_px/2)
    w = torch.full_like(cx, float(patch_px))
    h = torch.full_like(cy, float(patch_px))
    return torch.stack([cx, cy, w, h], dim=1)  # [N,4]

def _random_boxes_multiscale(H: int, W: int,
                             patch_fracs: Sequence[float],
                             counts: Sequence[int],
                             min_px: int,
                             device=None, dtype=None):
    """
    Random boxes at multiple scales, expressed as fractions of min(H,W).
    Ensures min side >= min_px.
    """
    assert len(patch_fracs) == len(counts)
    MN = min(H, W)
    boxes = []
    for f, n in zip(patch_fracs, counts):
        size = max(int(round(MN * float(f))), min_px)
        if size > min(H, W):
            size = min(H, W)
        # sample centers uniformly feasible for this size
        cx = torch.randint(low=size//2, high=W - size//2 + 1, size=(n,), device=device, dtype=torch.long)
        cy = torch.randint(low=size//2, high=H - size//2 + 1, size=(n,), device=device, dtype=torch.long)
        # Convert to float with proper dtype
        cx = cx.to(dtype=dtype)
        cy = cy.to(dtype=dtype)
        w  = torch.full_like(cx, float(size))
        h  = torch.full_like(cy, float(size))
        boxes.append(torch.stack([cx, cy, w, h], dim=1))
    return torch.cat(boxes, dim=0) if boxes else torch.empty(0,4, device=device, dtype=dtype)

def _clip_diy_weights(H: int, W: int, patch_fracs: Sequence[float], device=None, dtype=None) -> torch.Tensor:
    """
    Heuristic per-scale weights that:
      * favor global at low res,
      * balanced at mid res,
      * favor local at high res.
    Returns per-frac weights (sum to 1).
    """
    mn = min(H, W)
    # normalize ~ [0,1] where 0: <=256, 1: >=2048 (wider range for more gradual transition)
    t = float(mn - 75) / float(max(3500 - 75, 1))
    t = max(0.0, min(1.0, t))

    # base weights per scale index (0=global largest frac, last=local smallest)
    n = len(patch_fracs)
    idx = torch.arange(n, device=device, dtype=dtype)
    
    # More gradual transition: use softer scaling and preserve global signal longer
    # At low res (t=0): heavily favor global
    # At mid res (t=0.5): balanced
    # At high res (t=1): favor local but still keep some global
    w_global = 1.0 
    w_local  = torch.softmax(idx * (0.3 + 0.7*t), dim=0)     # softer scaling
    
    # Adaptive blending: preserve more global signal at higher resolutions
    # At low res: 70% global, 30% local
    # At high res: 30% global, 70% local (instead of 50/50)
    global_weight = 0.9 * (1.0 - t) + 0.1 * t
    local_weight = 1.0 - global_weight
    
    w = global_weight * w_global + local_weight * w_local
    return (w / w.sum()).detach()

def _seamless_edges_loss(x_img: torch.Tensor) -> torch.Tensor:
    """Encourage wrap-around continuity."""
    y = x_img.unsqueeze(0) if x_img.dim() == 3 else x_img
    return 0.5 * (F.mse_loss(y[:, :, :, 0], y[:, :, :, -1]) + 
                  F.mse_loss(y[:, :, 0, :], y[:, :, -1, :]))

def _down_up(x: torch.Tensor, side: int) -> torch.Tensor:
    """Downscale then upscale to kill high-frequency detail."""
    return F.interpolate(
        F.interpolate(x, size=(side, side), mode='bilinear', align_corners=False),
        size=x.shape[-2:], mode='bilinear', align_corners=False
    )

def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Cheap isotropic Gaussian blur via separable conv; falls back to avgpool if sigma<=0."""
    if sigma <= 0:
        return img
    # kernel size ~ 6*sigma rounded up to odd
    k = max(3, int(2 * round(3.0 * sigma) + 1))
    x = torch.arange(k, device=img.device) - (k - 1) / 2.0
    kern1d = torch.exp(-0.5 * (x / sigma)**2)
    kern1d = (kern1d / kern1d.sum()).to(img.dtype)
    ky = kern1d.view(1, 1, k, 1)
    kx = kern1d.view(1, 1, 1, k)
    pad = (k // 2)
    # depthwise conv
    C = img.shape[1]
    img = F.conv2d(img, ky.expand(C, 1, k, 1), padding=(pad, 0), groups=C)
    img = F.conv2d(img, kx.expand(C, 1, 1, k), padding=(0, pad), groups=C)
    return img

def _contrast_regularization(x_img: torch.Tensor) -> torch.Tensor:
    """Encourage black/white extremes by penalizing middle-gray values."""
    return (1.0 - torch.abs(x_img - 0.5)).mean()

def _to_clip_rgb(x_gray: torch.Tensor) -> torch.Tensor:
    """Convert single channel to RGB for CLIP."""
    return x_gray.repeat(1, 3, 1, 1)

@torch.no_grad()
def _load_clip_model(clip_model_name, device):
    clip_model, transforms = clip.load(clip_model_name, device=device, jit=False)
    return clip_model.eval().requires_grad_(False), transforms

@torch.no_grad()
def _encode_texts(clip_model, texts, device):
    """Encode multiple texts using CLIP model. Returns unit-normalized [K,D]."""
    if isinstance(texts, str):
        texts = [texts]
    tokens = clip.tokenize(texts).to(device)
    z = clip_model.encode_text(tokens).float()
    return F.normalize(z, dim=-1)

@torch.no_grad()
def _encode_image_microbatch(encoder: nn.Module, imgs: torch.Tensor, chunk: int = 32, device: torch.device | None = None):
    """Encode images in microbatches to manage memory usage."""
    device = device or imgs.device
    outs = []
    
    for i in range(0, imgs.size(0), chunk):
        batch = imgs[i:i+chunk]
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outs.append(encoder(batch).float())
        else:
            outs.append(encoder(batch).float())
    
    return torch.cat(outs, dim=0)

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in NCHW format, keeping single channel."""
    if x.dim() == 2:
        return x[None, None]
    elif x.dim() == 3:
        if x.shape[0] == 1:
            return x[None]
        raise ValueError("3D tensor must be (C,H,W) with C=1.")
    elif x.dim() != 4:
        raise ValueError("Input must be (H,W)/(C,H,W)/(N,C,H,W).")
    return x

def _to_clip_rgb(x: torch.Tensor) -> torch.Tensor:
    """Convert single channel to RGB for CLIP."""
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

# ---- helpers for conv-feature distances ----
def _flatten_norm_feats(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t.view(t.size(0), -1), dim=1)

def L2_layers(x_feats, y_feats):
    return [F.mse_loss(_flatten_norm_feats(a), _flatten_norm_feats(b)) for a, b in zip(x_feats, y_feats)]

def L1_layers(x_feats, y_feats):
    return [F.l1_loss(_flatten_norm_feats(a), _flatten_norm_feats(b)) for a, b in zip(x_feats, y_feats)]

# ---- text/image preprocessing consistent with CLIP ----
def _preprocess_image_for_clip(imgs: torch.Tensor, input_res: int, mean: torch.Tensor, std: torch.Tensor):
    imgs = _ensure_nchw(imgs)
    imgs = _to_clip_rgb(imgs)  # Convert to RGB right before CLIP
    if imgs.shape[-2] != input_res or imgs.shape[-1] != input_res:
        imgs = F.interpolate(imgs, size=(input_res, input_res), mode='bilinear', align_corners=False)
    imgs = imgs.to(mean.device, non_blocking=True)
    return (imgs.sub_(mean).div_(std)).float()

def _pairwise_cosine_spread(z_img: torch.Tensor, weights: torch.Tensor | None = None):
    """
    O(N·D) version of 1 - mean pairwise cosine for unit-norm rows of z_img.
    If weights is provided, it must be non-negative and will be normalized.
    """
    N, D = z_img.shape
    if N <= 1:
        return z_img.new_tensor(0.0)

    if weights is None:
        s = z_img.sum(dim=0)                         # [D]
        mean_pair_cos = (s.dot(s) - N) / (N * (N - 1))
        return 1.0 - mean_pair_cos

    w = (weights / weights.sum().clamp_min(1e-8)).to(z_img)
    s = (w[:, None] * z_img).sum(dim=0)              # [D]
    sum_w2 = (w * w).sum()
    denom = (1.0 - sum_w2).clamp_min(1e-8)
    mean_pair_cos = (s.dot(s) - sum_w2) / denom
    return 1.0 - mean_pair_cos

@dataclass(frozen=True)
class WeightedPrompt:
    text: str
    weight: float = 1.0

def _combine_weighted_prompts(embeds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Combine weighted prompts into single unit vector."""
    w = (weights / weights.sum().clamp_min(1e-8)).view(-1, 1)
    return F.normalize((w * embeds).sum(dim=0, keepdim=True), dim=-1)[0]

class PairedCrops(nn.Module):
    def __init__(self, crop_size, num_augs, min_original_size=480, noise=0.0, paired_noise=True):
        super().__init__()
        self.crop_size = int(crop_size)
        self.num_augs = int(num_augs)
        self.noise = float(noise)
        self.paired_noise = bool(paired_noise)

        scale_min = crop_size / float(min_original_size)
        try:
            rrc = K.RandomResizedCrop(size=(crop_size, crop_size),
                                        scale=(scale_min, 1.0),
                                        cropping_mode="resample")
        except TypeError:
            rrc = K.RandomResizedCrop(size=(crop_size, crop_size),
                                        scale=(scale_min, 1.0))

        ops = [
            rrc,
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        ]

        self.augs_single = K.AugmentationSequential(
            *ops, data_keys=["input"], same_on_batch=False, random_apply=False
        )
        self.augs_paired = K.AugmentationSequential(
            *ops, data_keys=["input", "input"], same_on_batch=False, random_apply=False
        )

    def _repeat(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_nchw(x)
        return x.repeat(self.num_augs, 1, 1, 1).contiguous(memory_format=torch.channels_last)

    def single(self, image: torch.Tensor) -> torch.Tensor:
        x_rep = self._repeat(image)
        x_aug = self.augs_single(x_rep)
        if self.noise > 0:
            facs = x_aug.new_empty([self.num_augs, 1, 1, 1]).uniform_(0, self.noise)
            x_aug = x_aug + facs * torch.randn_like(x_aug)
        return x_aug

    def paired(self, image: torch.Tensor, target: torch.Tensor):
        x_rep = self._repeat(image)
        y_rep = self._repeat(target)
        x_aug, y_aug = self.augs_paired(x_rep, y_rep)
        if self.noise > 0:
            facs = x_aug.new_empty([self.num_augs, 1, 1, 1]).uniform_(0, self.noise)
            eps  = torch.randn_like(x_aug)
            if self.paired_noise:
                x_aug = x_aug + facs * eps
                y_aug = y_aug + facs * eps
            else:
                x_aug = x_aug + facs * eps
                y_aug = y_aug + facs * torch.randn_like(y_aug)
        return x_aug, y_aug


class RNTrunk(nn.Module):
    """
    Wraps CLIP-ResNet visual trunk, exposing stem + layers 1..4 + attnpool.
    Returns pooled embedding and a list of feature maps [x0,x1,x2,x3,x4].
    """
    def __init__(self, rn_visual: nn.Module, device: torch.device):
        super().__init__()
        if not hasattr(rn_visual, "attnpool"):
            raise ValueError("clip_rn_model_name must be a CLIP ResNet variant (has .attnpool).")
        self.vis = rn_visual
        self.device = device

    def _stem(self, m, x):
        for conv, bn, relu in [(m.conv1, m.bn1, m.relu1),
                               (m.conv2, m.bn2, m.relu2),
                               (m.conv3, m.bn3, m.relu3)]:
            x = relu(bn(conv(x)))
        return m.avgpool(x)

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=self.vis.conv1.weight.dtype)
        x0 = self._stem(self.vis, x)
        x1 = self.vis.layer1(x0)
        x2 = self.vis.layer2(x1)
        x3 = self.vis.layer3(x2)
        x4 = self.vis.layer4(x3)
        y  = self.vis.attnpool(x4)
        return y, [x0, x1, x2, x3, x4]

class CLIPEncode(nn.Module):
    def __init__(self, clip_model): super().__init__(); self.m = clip_model
    def forward(self, x): return self.m.encode_image(x)

class CLIPLoss(nn.Module):
    """
    CLIP guidance loss with paired crops. Initialize with prompts + weights.
      - If image_prompt is provided -> image-to-image conv-feature loss.
      - Else -> image-to-text loss vs combined positive (and optional negatives).
    """
    def __init__(
        self,
        clip_model_name,
        clip_rn_model_name,
        *,
        device,
        positive_prompts=None,
        pos_weights=None,                 # weights for positive_prompts
        negative_prompts=None,
        neg_weights=None,                 # weights for negative_prompts
        image_prompt=None,
        num_augs: int = 32,
        conv_loss_type: str = "L2",
        conv_layer_weights = (0.2, 0.4, 0.6, 0.8, 0.2),
        use_arcsin_transform: bool = True,
        margin: float = 0.1,
        # multi-patch pyramid and tiling controls
        use_patch_pyramid: bool = True,
        patch_fracs: Tuple[float,...] = (1.00, 0.75, 0.50),   # relative to min(H,W)
        crops_per_frac: Tuple[int,...] = (4, 8, 16),
        min_patch_px: int = 96,
        hard_mining_frac: float = 0.25,     # weight top-k hardest patches more
        fixed_grid_at_highres: bool = True,  # switch to fixed tiling for large images
        tiled_min_res: int = 450,            # if min(H,W) >= this, enable tiling
        tile_px: int = 336,                   # tile size at input scale
        tile_overlap: float = 0.33,           # 0..0.9
        tile_jitter_frac: float = 0.05,       # small jitter for tiles
        # global/composition path
        use_global_path: bool = True,
        global_downside: int = 100,
        center_crop_frac: float = 0.90,
        num_global_views: int = 2,
        # low frequency regularization (fade out as resolution increases)
        use_lowfreq_reg: bool = True,
        lowfreq_sigma: float = 5.0,
        lowfreq_weight_max: float = 0.5,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_augs = int(num_augs)
        self.conv_loss_type = conv_loss_type
        self.conv_layer_weights = list(conv_layer_weights)
        self.use_arcsin_transform = bool(use_arcsin_transform)
        self.margin = float(margin)
        self.use_pairwise_spread = True
        self.consistency_weight = 0.03

        # Multi-patch pyramid and tiling parameters
        self.use_patch_pyramid = use_patch_pyramid
        self.patch_fracs = tuple(patch_fracs)
        self.crops_per_frac = tuple(crops_per_frac)
        self.min_patch_px = int(min_patch_px)
        self.hard_mining_frac = float(hard_mining_frac)
        self.fixed_grid_at_highres = bool(fixed_grid_at_highres)
        self.tiled_min_res = int(tiled_min_res)
        self.tile_px = int(tile_px)
        self.tile_overlap = float(tile_overlap)
        self.tile_jitter_frac = float(tile_jitter_frac)

        # Global/composition parameters
        self.use_global_path = bool(use_global_path)
        self.global_downside = int(global_downside)
        self.center_crop_frac = float(center_crop_frac)
        self.num_global_views = int(num_global_views)

        # Low frequency regularization
        self.use_lowfreq_reg = bool(use_lowfreq_reg)
        self.lowfreq_sigma = float(lowfreq_sigma)
        self.lowfreq_weight_max = float(lowfreq_weight_max)

        # Models
        self.clip_model, _    = _load_clip_model(clip_model_name, device=self.device)
        self.clip_rn_model, _ = _load_clip_model(clip_rn_model_name, device=self.device)

        # RN trunk and encoder (compilable)
        self.rn_trunk = RNTrunk(self.clip_rn_model.visual, self.device).to(self.device).eval()
        self.clip_encoder = CLIPEncode(self.clip_model).eval().to(self.device)
        if hasattr(torch, "compile"):
            try:
                self.rn_trunk = torch.compile(self.rn_trunk, mode="default", fullgraph=False)
            except Exception:
                pass  # safe fallback

        # Cropper with paired mode
        self.clip_input_res = int(self.clip_model.visual.input_resolution)
        self.cropper = PairedCrops(
            self.clip_input_res, 
            self.num_augs, 
            min_original_size=480, 
            noise=0.0, 
            paired_noise=True,
        ).to(self.device)

        # CLIP mean/std buffers
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

        self.distance_metrics = {"L2": L2_layers, "L1": L1_layers}
        if self.conv_loss_type not in self.distance_metrics:
            raise ValueError(f"conv_loss_type must be one of {list(self.distance_metrics.keys())}")

        # Normalize inputs to lists
        pos = [positive_prompts] if isinstance(positive_prompts, str) else (positive_prompts or [])
        neg = [negative_prompts] if isinstance(negative_prompts, str) else (negative_prompts or [])

        def _coerce_weights(prompts, w):
            if not prompts:
                return []
            if w is None:
                return [1.0] * len(prompts)
            if len(w) != len(prompts):
                raise ValueError("weights must match length of prompts")
            return [float(x) for x in w]

        w_pos = _coerce_weights(pos, pos_weights)
        w_neg = _coerce_weights(neg, neg_weights)

        # Store prompts and image target
        self.image_prompt = image_prompt
        self._E_neg_bank = None  # Store individual negative embeddings
        self._set_and_cache_prompts(pos, w_pos, neg, w_neg)

    # ---- prompt handling (simple & explicit) ----
    def _make_weighted(self, texts, weights):
        return [WeightedPrompt(t, w) for t, w in zip(texts, weights)]

    def _encode_weighted(self, prompts_w):
        """Encode and combine weighted prompts into single unit vector."""
        if not prompts_w:
            return None
        texts = [p.text for p in prompts_w]
        weights = torch.tensor([p.weight for p in prompts_w], device=self.device)
        embeds = _encode_texts(self.clip_model, texts, self.device)
        return _combine_weighted_prompts(embeds, weights)

    def _set_and_cache_prompts(self, pos_texts, pos_weights, neg_texts, neg_weights):
        self.pos_prompts = self._make_weighted(pos_texts, pos_weights) if pos_texts else []
        self.neg_prompts = self._make_weighted(neg_texts, neg_weights) if neg_texts else []
        
        # Cache combined embeddings
        self.e_pos = self._encode_weighted(self.pos_prompts)
        self.e_neg = self._encode_weighted(self.neg_prompts)
        
        # Store individual negative embeddings for better negative guidance
        if self.neg_prompts:
            texts = [p.text for p in self.neg_prompts]
            self._E_neg_bank = _encode_texts(self.clip_model, texts, self.device)
        else:
            self._E_neg_bank = None

    @torch.no_grad()
    def set_prompts(self, positive_prompts=None, pos_weights=None, negative_prompts=None, neg_weights=None):
        """Runtime update of prompts; same rules as __init__."""
        pos = [positive_prompts] if isinstance(positive_prompts, str) else (positive_prompts or [])
        neg = [negative_prompts] if isinstance(negative_prompts, str) else (negative_prompts or [])

        def _coerce(prompts, w):
            if not prompts:
                return [], []
            if w is None:
                return prompts, [1.0] * len(prompts)
            if len(w) != len(prompts):
                raise ValueError("weights must match length of prompts")
            return prompts, [float(x) for x in w]

        pos, w_pos = _coerce(pos, pos_weights)
        neg, w_neg = _coerce(neg, neg_weights)
        self._set_and_cache_prompts(pos, w_pos, neg, w_neg)

    # ---- distance on the unit sphere or cosine ----
    def _spherical_loss(self, z_img, z_txt, use_arcsin=True):
        """Compute spherical or cosine distance loss."""
        if z_txt is None:
            return z_img.new_tensor(0.0)
        
        if use_arcsin:
            d = torch.norm(z_img - z_txt[None, :], dim=1).clamp(0.0, 2.0 - 1e-6)
            return (torch.arcsin(d * 0.5) ** 2).mean()
        
        cos = (z_img * z_txt[None, :]).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
        return (1.0 - cos).mean()

    # ---- geometry: image↔image via conv features with paired augs ----
    def evaluate_image_to_image(self, input_image: torch.Tensor, target_image: torch.Tensor):
        x_aug, y_aug = self.cropper.paired(_ensure_nchw(input_image), _ensure_nchw(target_image))

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, x_feats = self.rn_trunk(x_aug)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, y_feats = self.rn_trunk(y_aug)
        else:
            _, x_feats = self.rn_trunk(x_aug)
            with torch.inference_mode():
                _, y_feats = self.rn_trunk(y_aug)

        conv_losses = self.distance_metrics[self.conv_loss_type](x_feats, y_feats)
        total = x_aug.new_tensor(0.0)
        for idx, w in enumerate(self.conv_layer_weights):
            if w:
                total = total + float(w) * conv_losses[idx]
        
        # Add pairwise cosine spread loss if enabled
        if self.use_pairwise_spread:
            imgs = _preprocess_image_for_clip(x_aug, self.clip_input_res, self.clip_mean, self.clip_std)
            z_img = F.normalize(_encode_image_microbatch(self.clip_encoder, imgs, chunk=32, device=self.device), dim=-1)
            total = total + self.consistency_weight * _pairwise_cosine_spread(z_img, weights=None)
            
        return total

    # ---- semantics: image↔text (positives + optional negatives with hinge) ----
    def evaluate_image_to_text(self, input_image: torch.Tensor, use_arcsin_transform: bool = True):
        x = _ensure_nchw(input_image)
        B, C, H, W = x.shape
        assert B == 1, "Assumes single image."

        crops = []

        # ---------- (A) GLOBAL VIEWS FOR COMPOSITION ----------
        # Always include full image (no jitter), plus a low-passed view via down-up.
        if self.use_global_path:
            # Full-image resize to CLIP input (no aug)
            full = F.interpolate(x, size=(self.clip_input_res, self.clip_input_res),
                                 mode='bilinear', align_corners=False)
            crops.append(full)
            # Down-up view focuses gradients on global layout
            downup = _down_up(x, side=self.global_downside)
            downup = F.interpolate(downup, size=(self.clip_input_res, self.clip_input_res),
                                   mode='bilinear', align_corners=False)
            crops.append(downup)
            # Optional: big center crop to de-emphasize borders
            if self.num_global_views >= 3 and self.center_crop_frac < 1.0:
                frac = max(0.1, min(1.0, self.center_crop_frac))
                ph = int(round(min(H, W) * frac))
                pw = ph
                cx = W // 2; cy = H // 2
                x0 = max(0, cx - pw // 2); y0 = max(0, cy - ph // 2)
                x1 = min(W, x0 + pw); y1 = min(H, y0 + ph)
                center = F.interpolate(x[:, :, y0:y1, x0:x1],
                                       size=(self.clip_input_res, self.clip_input_res),
                                       mode='bilinear', align_corners=False)
                crops.append(center)

        # ---------- (B) MULTI-SCALE (CLIP-DIY) RANDOM PATCH PYRAMID ----------
        if self.use_patch_pyramid:
            boxes_rand = _random_boxes_multiscale(
                H, W,
                patch_fracs=self.patch_fracs,
                counts=self.crops_per_frac,
                min_px=self.min_patch_px,
                device=x.device,
                dtype=x.dtype
            )
            if boxes_rand.numel() > 0:
                grid_rand = _affine_grid_from_boxes(H, W, boxes_rand,
                                                    (self.clip_input_res, self.clip_input_res))
                x_rep = x.repeat(boxes_rand.size(0), 1, 1, 1)
                crops_rand = F.grid_sample(x_rep, grid_rand, mode='bilinear', align_corners=False)
                crops.append(crops_rand)

        # ---------- (C) HIGH-RES FIXED TILES ----------
        mn = min(H, W)
        if self.fixed_grid_at_highres and mn >= self.tiled_min_res:
            stride = max(1, int(round(self.tile_px * (1.0 - self.tile_overlap))))
            boxes_fix = _fixed_grid_boxes(H, W, patch_px=self.tile_px, stride_px=stride,
                                          jitter_frac=self.tile_jitter_frac, device=x.device, dtype=x.dtype)
            if boxes_fix.numel() > 0:
                grid_fix = _affine_grid_from_boxes(H, W, boxes_fix,
                                                   (self.clip_input_res, self.clip_input_res))
                x_rep = x.repeat(boxes_fix.size(0), 1, 1, 1)
                crops_fix = F.grid_sample(x_rep, grid_fix, mode='bilinear', align_corners=False)
                crops.append(crops_fix)

        if not crops:
            crops = [self.cropper.single(x)]
        imgs = torch.cat(crops, dim=0)

        # ---------- CLIP ENCODE ----------
        imgs = _preprocess_image_for_clip(imgs, self.clip_input_res, self.clip_mean, self.clip_std)
        
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_img = F.normalize(self.clip_encoder(imgs).float(), dim=-1)
        else:
            z_img = F.normalize(self.clip_encoder(imgs).float(), dim=-1)

        # ---------- POSITIVE LOSS (weighted by scale buckets) ----------
        if self.e_pos is None:
            L_pos = z_img.new_tensor(0.0)
        else:
            if use_arcsin_transform:
                d = torch.norm(z_img - self.e_pos[None, :], dim=1).clamp(0.0, 2.0 - 1e-6)
                per_crop = (torch.arcsin(d * 0.5) ** 2)
            else:
                cos = (z_img * self.e_pos[None, :]).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
                per_crop = (1.0 - cos)

            # Build per-crop weights: start with ones
            weights = torch.ones_like(per_crop)

            # Give global views extra weight (appended first)
            offset = 0
            if self.use_global_path:
                n_global = 2 + (1 if (self.num_global_views >= 3 and self.center_crop_frac < 1.0) else 0)
                # Global weight ramps from ~2.5 at low res down to ~1.0 at high res
                t = float(mn - 256) / float(max(1024 - 256, 1))
                t = max(0.0, min(1.0, t))
                global_w = 2.5 - 1.5 * t
                weights[offset:offset + n_global] = global_w
                offset += n_global

            # Patch pyramid weights (as before)
            if self.use_patch_pyramid:
                scale_w = _clip_diy_weights(H, W, self.patch_fracs, device=per_crop.device, dtype=per_crop.dtype)
                for i, cnt in enumerate(self.crops_per_frac):
                    if cnt <= 0:
                        continue
                    weights[offset : offset + cnt] = weights[offset : offset + cnt] * scale_w[i]
                    offset += cnt

            # Tiles: keep equal to mean of previous weights
            if self.fixed_grid_at_highres and mn >= self.tiled_min_res:
                tile_cnt = per_crop.numel() - offset
                if tile_cnt > 0 and offset > 0:
                    mean_w = weights[:offset].mean()
                    weights[offset : offset + tile_cnt] = mean_w

            # Hard mining on non-global crops only
            if self.hard_mining_frac > 0:
                start = n_global if self.use_global_path else 0
                if per_crop.numel() - start > 0:
                    k = max(1, int(round(self.hard_mining_frac * (per_crop.numel() - start))))
                    _, idxs = torch.topk(per_crop[start:], k=k, largest=True, sorted=False)
                    weights[start + idxs] *= 2.0

            L_pos = (weights * per_crop).sum() / weights.sum().clamp_min(1e-8)

        # ---------- NEGATIVE (hinge) ----------
        L_neg = z_img.new_tensor(0.0)
        if self._E_neg_bank is not None and self._E_neg_bank.numel() > 0:
            sims_neg = z_img @ self._E_neg_bank.t()
            cos_neg_max = sims_neg.max(dim=1).values
            cos_pos = (z_img * self.e_pos[None, :]).sum(dim=1) if self.e_pos is not None else z_img.new_zeros(z_img.size(0))
            hinge = F.relu(self.margin + cos_neg_max - cos_pos)
            L_neg = (weights * hinge).sum() / weights.sum().clamp_min(1e-8)

        # ---------- LOW-FREQUENCY REGULARIZER (early → fade out) ----------
        L_low = z_img.new_tensor(0.0)
        if self.use_lowfreq_reg:
            # Stronger when small, fades as resolution grows
            t = float(mn - 256) / float(max(1024 - 256, 1))
            t = max(0.0, min(1.0, t))
            w_low = (1.0 - t) * self.lowfreq_weight_max
            if w_low > 0:
                L_low = F.l1_loss(x, _gaussian_blur(x, sigma=self.lowfreq_sigma)) * w_low

        # ---------- CONTRAST REGULARIZER (encourage black/white extremes) ----------
        L_contrast = _contrast_regularization(x) * self.contrast_weight if hasattr(self, 'use_contrast_reg') and self.use_contrast_reg else z_img.new_tensor(0.0)

        return L_pos + L_neg + L_low + L_contrast

    def forward(self, logits: torch.Tensor):
        input_image = torch.sigmoid(logits)
        loss = (self.evaluate_image_to_image(input_image, self.image_prompt) 
                if self.image_prompt is not None 
                else self.evaluate_image_to_text(input_image, self.use_arcsin_transform))
        return 100.0 * (loss + _seamless_edges_loss(input_image) * 0.15)