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

def _fixed_grid_boxes(H: int, W: int, patch_px: int, stride_px: int, jitter_frac: float = 0.0, device=None):
    """
    Create a fixed, overlapping grid of (cx, cy, w, h) boxes in pixels covering the image.
    patch_px: square crop (w=h=patch_px at input scale)
    stride_px: stride between patch centers
    jitter_frac: random jitter up to +/- jitter_frac * stride_px on centers (per patch)
    """
    xs = torch.arange(patch_px//2, W - patch_px//2 + 1, stride_px, device=device)
    ys = torch.arange(patch_px//2, H - patch_px//2 + 1, stride_px, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    cx = gx.reshape(-1).float()
    cy = gy.reshape(-1).float()
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
                             device=None):
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
        cx = torch.randint(low=size//2, high=W - size//2 + 1, size=(n,), device=device).float()
        cy = torch.randint(low=size//2, high=H - size//2 + 1, size=(n,), device=device).float()
        w  = torch.full_like(cx, float(size))
        h  = torch.full_like(cy, float(size))
        boxes.append(torch.stack([cx, cy, w, h], dim=1))
    return torch.cat(boxes, dim=0) if boxes else torch.empty(0,4, device=device)

def _clip_diy_weights(H: int, W: int, patch_fracs: Sequence[float]) -> torch.Tensor:
    """
    Heuristic per-scale weights that:
      * favor global at low res,
      * balanced at mid res,
      * favor local at high res.
    Returns per-frac weights (sum to 1).
    """
    mn = min(H, W)
    # normalize ~ [0,1] where 0: <=256, 1: >=1024
    t = float(mn - 256) / float(max(1024 - 256, 1))
    t = max(0.0, min(1.0, t))
    # base weights per scale index (0=global largest frac, last=local smallest)
    n = len(patch_fracs)
    idx = torch.arange(n).float()
    # linear ramp from global-heavy to local-heavy across t
    w_global = torch.softmax(-(idx)* (0.5 + 1.0*t), dim=0)
    w_local  = torch.softmax(idx * (0.5 + 1.0*t), dim=0)
    # blend to keep some global signal at high res and some local at low res
    w = 0.5 * w_global + 0.5 * w_local
    return (w / w.sum()).detach()

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
    if device.type == "cuda":
        for i in range(0, imgs.size(0), chunk):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outs.append(encoder(imgs[i:i+chunk]).float())
    else:
        for i in range(0, imgs.size(0), chunk):
            outs.append(encoder(imgs[i:i+chunk]).float())
    return torch.cat(outs, dim=0)

def _ensure_nchw3(x: torch.Tensor):
    if x.dim() == 2:        # H,W
        x = x[None, None]
    elif x.dim() == 3:      # C,H,W
        if x.shape[0] in (1, 3):
            x = x[None]
        else:
            raise ValueError("3D tensor must be (C,H,W) with C in {1,3}.")
    elif x.dim() != 4:
        raise ValueError("Input must be (H,W)/(C,H,W)/(N,C,H,W).")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x

# ---- helpers for conv-feature distances ----
def _flatten_norm_feats(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t.view(t.size(0), -1), dim=1)

def L2_layers(x_feats, y_feats):
    return [F.mse_loss(_flatten_norm_feats(a), _flatten_norm_feats(b)) for a, b in zip(x_feats, y_feats)]

def L1_layers(x_feats, y_feats):
    return [F.l1_loss(_flatten_norm_feats(a), _flatten_norm_feats(b)) for a, b in zip(x_feats, y_feats)]

# ---- text/image preprocessing consistent with CLIP ----
def _preprocess_image_for_clip(imgs: torch.Tensor, input_res: int, mean: torch.Tensor, std: torch.Tensor):
    imgs = _ensure_nchw3(imgs)
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

def _combine_weighted_prompts(embeds: torch.Tensor, weights: torch.Tensor):
    """embeds: [K,D] unit-norm, weights: [K] -> returns [D] unit-norm"""
    w = (weights / weights.sum().clamp_min(1e-8)).view(-1, 1)
    z = (w * embeds).sum(dim=0, keepdim=True)
    return F.normalize(z, dim=-1)[0]  # [D]

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
        x = _ensure_nchw3(x)
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
        conv_layer_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
        use_arcsin_transform: bool = True,
        margin: float = 0.1,
        # NEW: multi-patch pyramid and tiling controls
        use_patch_pyramid: bool = True,
        patch_fracs: Tuple[float,...] = (1.00, 0.50, 0.25),   # relative to min(H,W)
        crops_per_frac: Tuple[int,...] = (4, 8, 16),
        min_patch_px: int = 96,
        hard_mining_frac: float = 0.25,     # weight top-k hardest patches more
        fixed_grid_at_highres: bool = True,  # switch to fixed tiling for large images
        tiled_min_res: int = 640,            # if min(H,W) >= this, enable tiling
        tile_px: int = 336,                   # tile size at input scale
        tile_overlap: float = 0.33,           # 0..0.9
        tile_jitter_frac: float = 0.05,       # small jitter for tiles
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
        self.patch_fracs       = tuple(patch_fracs)
        self.crops_per_frac    = tuple(crops_per_frac)
        self.min_patch_px      = int(min_patch_px)
        self.hard_mining_frac  = float(hard_mining_frac)
        self.fixed_grid_at_highres = bool(fixed_grid_at_highres)
        self.tiled_min_res     = int(tiled_min_res)
        self.tile_px           = int(tile_px)
        self.tile_overlap      = float(tile_overlap)
        self.tile_jitter_frac  = float(tile_jitter_frac)

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

        # Coerce weights
        def _coerce_weights(prompts, w):
            if prompts is None or len(prompts) == 0:
                return []
            if w is None:
                return [1.0] * len(prompts)
            if len(w) != len(prompts):
                raise ValueError("pos_weights must match length of positive_prompts")
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
        """Encode and combine a list[WeightedPrompt] into a single unit vector [D]."""
        if not prompts_w:
            return None
        E = _encode_texts(self.clip_model, [p.text for p in prompts_w], self.device)  # [K,D]
        W = torch.tensor([p.weight for p in prompts_w], device=E.device, dtype=E.dtype)
        return _combine_weighted_prompts(E, W)  # [D]

    def _set_and_cache_prompts(self, pos_texts, pos_weights, neg_texts, neg_weights):
        self.pos_prompts = self._make_weighted(pos_texts, pos_weights) if pos_texts else []
        self.neg_prompts = self._make_weighted(neg_texts, neg_weights) if neg_texts else []
        # Cache combined embeddings (single vectors)
        self.e_pos = self._encode_weighted(self.pos_prompts)  # [D] or None
        self.e_neg = self._encode_weighted(self.neg_prompts)  # [D] or None
        # Store individual negative embeddings for better negative guidance
        self._E_neg_bank = _encode_texts(self.clip_model, [p.text for p in self.neg_prompts], self.device) if self.neg_prompts else None  # [K,D]

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
                raise ValueError("pos_weights must match length of prompts")
            return prompts, [float(x) for x in w]

        pos, w_pos = _coerce(pos, pos_weights)
        neg, w_neg = _coerce(neg, neg_weights)
        self._set_and_cache_prompts(pos, w_pos, neg, w_neg)

    # ---- distance on the unit sphere or cosine ----
    def _spherical_loss(self, z_img, z_txt, use_arcsin=True):
        """z_img: [B,D] unit; z_txt: [D] unit"""
        if z_txt is None:
            return z_img.new_tensor(0.0)
        if use_arcsin:
            d = torch.norm(z_img - z_txt[None, :], dim=1)
            d = torch.clamp(d, 0.0, 2.0 - 1e-6)
            return (torch.arcsin(d * 0.5) ** 2).mean()
        # cosine distance path
        cos = (z_img * z_txt[None, :]).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
        return (1.0 - cos).mean()

    # ---- geometry: image↔image via conv features with paired augs ----
    def evaluate_image_to_image(self, input_image: torch.Tensor, target_image: torch.Tensor):
        x_aug, y_aug = self.cropper.paired(_ensure_nchw3(input_image), _ensure_nchw3(target_image))

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
            # For image-to-image, we can apply spread to both input and target embeddings
            # Here we'll use the input image embeddings for spread
            imgs = _preprocess_image_for_clip(x_aug, self.clip_input_res, self.clip_mean, self.clip_std)
            z_img = _encode_image_microbatch(self.clip_encoder, imgs, chunk=32, device=self.device)
            z_img = F.normalize(z_img, dim=-1)
            spread_loss = _pairwise_cosine_spread(z_img, weights=None)
            total = total + self.consistency_weight * spread_loss
            
        return total

    # ---- semantics: image↔text (positives + optional negatives with hinge) ----
    def evaluate_image_to_text(self, input_image: torch.Tensor, use_arcsin_transform: bool = True):
        """
        Multi-scale CLIP-DIY style: combine random multi-frac crops (global/mid/local)
        and, for high-res images, a fixed overlapping tile coverage. Optionally hard-mines
        the worst-matching patches for stronger gradients.
        """
        x = _ensure_nchw3(input_image)
        B, C, H, W = x.shape
        assert B == 1, "This implementation assumes a single image tensor."

        crops = []

        # --- A) CLIP-DIY style random pyramid (global/mid/local) ---
        if self.use_patch_pyramid:
            boxes_rand = _random_boxes_multiscale(
                H, W,
                patch_fracs=self.patch_fracs,
                counts=self.crops_per_frac,
                min_px=self.min_patch_px,
                device=x.device
            )
            if boxes_rand.numel() > 0:
                grid_rand = _affine_grid_from_boxes(H, W, boxes_rand, (self.clip_input_res, self.clip_input_res))
                x_rep = x.repeat(boxes_rand.size(0), 1, 1, 1)
                crops_rand = F.grid_sample(x_rep, grid_rand, mode='bilinear', align_corners=False)
                crops.append(crops_rand)

        # --- B) Ultra-high-res: fixed overlapping tiles for coverage ---
        mn = min(H, W)
        if self.fixed_grid_at_highres and mn >= self.tiled_min_res:
            stride = max(1, int(round(self.tile_px * (1.0 - self.tile_overlap))))
            boxes_fix = _fixed_grid_boxes(H, W, patch_px=self.tile_px, stride_px=stride,
                                          jitter_frac=self.tile_jitter_frac, device=x.device)
            if boxes_fix.numel() > 0:
                grid_fix = _affine_grid_from_boxes(H, W, boxes_fix, (self.clip_input_res, self.clip_input_res))
                x_rep = x.repeat(boxes_fix.size(0), 1, 1, 1)
                crops_fix = F.grid_sample(x_rep, grid_fix, mode='bilinear', align_corners=False)
                crops.append(crops_fix)

        # Fallback: if nothing produced (tiny image), just use standard random crops
        if not crops:
            crops = [self.cropper.single(x)]
        imgs = torch.cat(crops, dim=0)  # [Ncuts,3,224,224] (or 336 if your model has that input_res)

        # CLIP preprocess (your function handles std/mean & resize if needed)
        imgs = _preprocess_image_for_clip(imgs, self.clip_input_res, self.clip_mean, self.clip_std)

        # Encode all crops with CLIP
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_img = self.clip_encoder(imgs).float()
        else:
            z_img = self.clip_encoder(imgs).float()
        z_img = F.normalize(z_img, dim=-1)  # [Ncuts, D]

        # Positive loss per-crop
        if self.e_pos is None:
            L_pos = z_img.new_tensor(0.0)
            Ncuts = z_img.size(0)
        else:
            # spherical (on unit sphere) or cosine; compute per-crop
            if use_arcsin_transform:
                d = torch.norm(z_img - self.e_pos[None, :], dim=1).clamp(0.0, 2.0 - 1e-6)
                per_crop = (torch.arcsin(d * 0.5) ** 2)  # [Ncuts]
            else:
                cos = (z_img * self.e_pos[None, :]).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
                per_crop = (1.0 - cos)

            # Optional scale-aware weighting (global vs local)
            # Build weights per-crop based on which frac bucket it came from.
            # For simplicity, infer counts by how we stacked crops above:
            weights = torch.ones_like(per_crop)
            offset = 0
            if self.use_patch_pyramid:
                # make per-scale weights
                scale_w = _clip_diy_weights(H, W, self.patch_fracs).to(per_crop.device)
                for i, cnt in enumerate(self.crops_per_frac):
                    if cnt <= 0: 
                        continue
                    weights[offset : offset + cnt] = scale_w[i]
                    offset += cnt
            if self.fixed_grid_at_highres and mn >= self.tiled_min_res:
                # tiles get equal per-patch weight but are many; normalize later
                tile_cnt = (per_crop.numel() - offset)
                if tile_cnt > 0:
                    weights[offset : offset + tile_cnt] = weights[offset : offset + tile_cnt] * (weights.mean() if offset > 0 else 1.0)

            # Hard mining: upweight top-k hardest patches
            if self.hard_mining_frac > 0:
                Ncuts = per_crop.numel()
                k = max(1, int(round(self.hard_mining_frac * Ncuts)))
                vals, idxs = torch.topk(per_crop, k=k, largest=True, sorted=False)
                # 2x weight on hardest, keep others at 1x of their current weight
                w = weights.clone()
                w[idxs] = w[idxs] * 2.0
                weights = w

            # Final positive loss: weighted mean
            L_pos = (weights * per_crop).sum() / (weights.sum().clamp_min(1e-8))
            Ncuts = per_crop.size(0)

        # Negative hinge per-crop (optional)
        L_neg = z_img.new_tensor(0.0)
        if self._E_neg_bank is not None and self._E_neg_bank.numel() > 0:
            sims_neg = z_img @ self._E_neg_bank.t()                    # [B,K]
            cos_neg_max = sims_neg.max(dim=1).values                   # [B]
            cos_pos = (z_img * self.e_pos[None, :]).sum(dim=1) if self.e_pos is not None else z_img.new_zeros(z_img.size(0), device=z_img.device, dtype=z_img.dtype)
            hinge = F.relu(self.margin + cos_neg_max - cos_pos)
            # Match positive weights if computed, else simple mean
            if 'weights' in locals():
                L_neg = (weights * hinge).sum() / (weights.sum().clamp_min(1e-8))
            else:
                L_neg = hinge.mean()

        # Pairwise cosine spread loss
        L_spread = z_img.new_tensor(0.0)
        if self.use_pairwise_spread:
            L_spread = _pairwise_cosine_spread(z_img, weights=None)  # equal weights for all crops

        return L_pos + L_neg + self.consistency_weight * L_spread

    def forward(self, logits: torch.Tensor):
        input_image = torch.sigmoid(logits)
        if self.image_prompt is not None:
            loss = self.evaluate_image_to_image(input_image, self.image_prompt)
        else:
            loss = self.evaluate_image_to_text(input_image, self.use_arcsin_transform)
        return 100.0 * loss