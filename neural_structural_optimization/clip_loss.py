# clip_loss_torch.py
import torch
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])   # RGB
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def _to_square(img):
    # img: [B, C, H, W] -> pad to square with white (1.0)
    B, C, H, W = img.shape
    size = max(H, W)
    pad_h = size - H
    pad_w = size - W
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # left, right, top, bottom
    return F.pad(img, pad, mode="constant", value=1.0)

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts: (H,W), (1,H,W), (B,H,W), (C,H,W), (B,H,W,C), (B,C,H,W)
    Returns: (B,C,H,W). Does NOT change dtype/device.
    """
    if x.dim() == 2:                 # (H,W)
        x = x.unsqueeze(0).unsqueeze(0)             # -> (1,1,H,W)
    elif x.dim() == 3:
        if x.shape[0] in (1, 3):     # (C,H,W)
            x = x.unsqueeze(0)                        # -> (1,C,H,W)
        elif x.shape[-1] in (1, 3):  # (H,W,C)
            x = x.permute(2,0,1).unsqueeze(0)         # -> (1,C,H,W)
        else:                         # (B,H,W)
            x = x.unsqueeze(1)                         # -> (B,1,H,W)
    elif x.dim() == 4:
        # If last dim looks like channels, assume NHWC and convert
        if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
            x = x.permute(0,3,1,2)                    # NHWC -> NCHW
        # else assume already NCHW (B,C,H,W)
    else:
        raise ValueError(f"Unsupported tensor dim {x.dim()} for CLIP path")

    return x

def _preprocess_for_clip(img01, device):
    """
    img01: [B,1,H,W] or [B,3,H,W], values in [0,1], torch tensor (requires_grad OK).
    Returns: [B,3,224,224], normalized for CLIP.
    """
    if img01.dim() != 4:
        raise ValueError("Expected [B,C,H,W]")
    B, C, H, W = img01.shape
    if C == 1:
        img01 = img01.repeat(1, 3, 1, 1)
    elif C != 3:
        raise ValueError("Channel count must be 1 or 3")

    x = _to_square(img01)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = _CLIP_MEAN.to(device)[None, :, None, None]
    std  = _CLIP_STD.to(device)[None, :, None, None]
    x = (x - mean) / std
    return x

class CLIPLoss(torch.nn.Module):
    def __init__(self, device="cuda", model_name="ViT-B/32",
                 weight = 0.1, target_text_prompt: str | None = None):
        super().__init__()
        self.device = torch.device(device)
        self.clip_model, _ = clip.load(model_name, device=self.device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.weight = weight
        self.target_text_prompt = target_text_prompt
        self.register_buffer("target_embed", None, persistent=False)
        if target_text_prompt is not None:
            self._set_text_target(target_text_prompt)

    @torch.no_grad()
    def _set_text_target(self, prompt: str):
        toks = clip.tokenize([prompt]).to(self.device)     # [1,77]
        feat = self.clip_model.encode_text(toks).float()   # [1,D]
        feat = F.normalize(feat, dim=-1)
        self.target_embed = feat
        self.target_text_prompt = prompt

    def get_text_loss(self, logits: torch.Tensor, prompt: str | None = None) -> torch.Tensor:
        """
        logits: [B,1,H,W] or [B,3,H,W], unconstrained (we map → [0,1])
        returns: per-sample loss [B] = 1 - cosine_similarity
        """
        if prompt is not None and prompt != self.target_text_prompt:
            self._set_text_target(prompt)

        # Keep CLIP path in fp32; physics can be float64.
        # Casting creates a new tensor but preserves grad.
        logits = _ensure_nchw(logits)
        design01 = torch.sigmoid(logits).float()

        x = _preprocess_for_clip(design01, self.device)          # [B,3,224,224], fp32
        img_feat = self.clip_model.encode_image(x).float()       # [B,D]
        img_feat = F.normalize(img_feat, dim=-1)

        target = self.target_embed                                # [1,D], fp32
        sim = (img_feat * target).sum(dim=-1)                     # [B]
        return 1.0 - sim                                          # [B]

    def forward(self, design01: torch.Tensor) -> torch.Tensor:
        """
        design01: [B,1,H,W] or [B,3,H,W], values in [0,1], requires_grad=True.
        Returns: scalar loss = weight * (1 - cosine_sim)
        """
        x = _preprocess_for_clip(design01.to(self.device).float(), self.device)
        img_feat = self.clip_model.encode_image(x).float()              # [B,D]
        img_feat = F.normalize(img_feat, dim=-1)

        # cosine similarity to target
        # broadcast target [1,D] against [B,D]
        sim = (img_feat * self.target_embed).sum(dim=-1)                # [B]
        loss = self.weight * (1.0 - sim).mean()
        return loss