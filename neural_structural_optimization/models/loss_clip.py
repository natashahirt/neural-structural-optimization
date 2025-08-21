from dataclasses import dataclass
from pandas.errors import LossySetitemError
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import clip

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
    def __init__(self, crop_size, num_augs, min_original_size=480, noise=0.0, paired_noise=True, use_fixed_crops=True, extra_random_crops=4):
        super().__init__()
        self.crop_size = int(crop_size)
        self.num_augs = int(num_augs)
        self.noise = float(noise)
        self.paired_noise = bool(paired_noise)
        self.use_fixed_crops = bool(use_fixed_crops)
        self.extra_random_crops = int(extra_random_crops)

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
        conv_layer_weights = (0.0, 1.0, 1.0, 1.0, 1.0),
        use_arcsin_transform: bool = True,
        margin: float = 0.1,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_augs = int(num_augs)
        self.conv_loss_type = conv_loss_type
        self.conv_layer_weights = list(conv_layer_weights)
        self.use_arcsin_transform = bool(use_arcsin_transform)
        self.margin = float(margin)

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
        return total

    # ---- semantics: image↔text (positives + optional negatives with hinge) ----
    def evaluate_image_to_text(self, input_image: torch.Tensor, use_arcsin_transform: bool = True):
        crops = self.cropper.single(_ensure_nchw3(input_image))
        imgs  = _preprocess_image_for_clip(crops, self.clip_input_res, self.clip_mean, self.clip_std)

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_img = self.clip_encoder(imgs).float()
        else:
            z_img = self.clip_encoder(imgs).float()
        z_img = F.normalize(z_img, dim=-1)

        # positive loss
        L_pos = self._spherical_loss(z_img, self.e_pos, use_arcsin_transform)

        # negative hinge (margin + cos_neg - cos_pos)
        L_neg = z_img.new_tensor(0.0)
        if self.e_neg is not None:
            cos_pos = (z_img * self.e_pos[None, :]).sum(dim=1) if self.e_pos is not None else z_img.new_zeros(z_img.size(0))
            cos_neg = (z_img * self.e_neg[None, :]).sum(dim=1)
            hinge = F.relu(self.margin + cos_neg - cos_pos)
            L_neg = hinge.mean()

        return L_pos + L_neg

    def forward(self, logits: torch.Tensor):
        input_image = torch.sigmoid(logits)
        input_image = 1.0 - input_image # invert
        if self.image_prompt is not None:
            loss = self.evaluate_image_to_image(input_image, self.image_prompt)
        else:
            loss = self.evaluate_image_to_text(input_image, self.use_arcsin_transform)
        return 100.0 * loss