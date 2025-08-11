# clip_loss_torch.py
import torch
import torch.nn.functional as F
from torchvision import transforms
import kornia.augmentation as K
import open_clip
from dataclasses import dataclass

def combine_prompts(prompts):
    if isinstance(prompts, list):
        return '_'.join(map(str, prompts))
    elif isinstance(prompts, str):
        return prompts
    else:
        raise TypeError("prompts must be of type list or str")

def encode_text(text, clip_model):
    tokenized_text = clip.tokenize(text).to(device)
    return clip_model.encode_text(tokenized_text).float()
    
class GenerateCrops(torch.nn.Module):
    def __init__(self, crop_size, batch_size, min_original_size=480, noise=0.1):
        super().__init__()

        self.batch_size = batch_size
        self.crop_size = crop_size
        self.min_original_size = min_original_size
        self.noise = noise

        self.scale_min = self.crop_size/self.min_original_size

        self.augs = torch.nn.Sequential(
          K.RandomResizedCrop(size=(self.crop_size, self.crop_size), scale=(self.scale_min, 1.0), cropping_mode ="resample"),
          K.RandomHorizontalFlip(p=0.5),
          K.RandomSharpness(0.3,p=0.4),
          K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
          K.RandomPerspective(0.2,p=0.4),
          K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
          )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        When an input image (or a batch of images) is passed to this module, 
        this method is called to perform the augmentations.
        """
        cutouts = torch.cat(self.batch_size*[input],dim=0) # Create a batch of identical images
        #  initializes several instance variables and creates an augmentation pipeline for the batch
        self.batch =  self.augs(cutouts)
        
        if self.noise > 0:
            facs = batch.new_empty([self.batch_size, 1, 1, 1]).uniform_(0, self.noise)
            self.batch = self.batch + facs * torch.randn_like(self.batch)
        return self.batch

class _Preprocess(torch.nn.Module):
    def __init__(self, device, size=224):
        super().__init__()
        self.size = size

        self.register_buffer("_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None])
        self.register_buffer("_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None])

    @staticmethod
    def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.shape[0] in (1, 3):          # (C,H,W)
                x = x.unsqueeze(0)
            elif x.shape[-1] in (1, 3):       # (H,W,C)
                x = x.permute(2, 0, 1).unsqueeze(0)
            else:                              # (B,H,W)
                x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
                x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported shape {tuple(x.shape)}")
        return x

    def _pad_to_square(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        S = max(H, W)
        canvas = img.new_empty((B, C, S, S))
        canvas[:] = self._mean
        top = (S - H) // 2
        left = (S - W) // 2
        canvas[:, :, top:top+H, left:left+W] = img
        return canvas

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._ensure_nchw(img)
        
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1) # greyscale to rgb

        x = x.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type=x.device.type, enabled=(x.device.type == 'cuda')):
            x = self._pad_to_square(x)
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
            x = (x - self._mean) / self._std

        x = x.to(dtype=torch.float32)
        return x.contiguous(memory_format=torch.channels_last)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device="auto", model_name="ViT-B/32", pretrained="openai",
                 weight=0.1, target_text_prompt: str | None = None):
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.clip_model = self.clip_model.to(self.device, dtype=torch.float32)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.weight = weight
        self.register_buffer("target_embed", None, persistent=False)
        self.target_text_prompt = None
        self._neg = None

        self.preprocess = _Preprocess(self.device, size=224)

        self._tokenizer = open_clip.get_tokenizer(model_name)

        if target_text_prompt is not None:
            self._set_text_target(target_text_prompt)

        # small global knobs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def set_negatives(self, prompts):
        # allow a single string or list of strings
        if isinstance(prompts, str):
            prompts = [prompts]
        toks = self._tokenizer(prompts).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            emb = self.clip_model.encode_text(toks).float()
        self._neg = F.normalize(emb, dim=-1).mean(0, keepdim=True)  # [1,D]

    @torch.no_grad()
    def _set_text_target(self, prompt: str):
        toks = self._tokenizer([prompt]).to(self.device)  
        feat = self.clip_model.encode_text(toks).float()
        self.target_embed = F.normalize(feat, dim=-1)  # [1,D]
        self.target_text_prompt = prompt

    def get_text_loss(self, logits, prompt: str | None = None, K=4, scale=(0.6, 1.0), jitter=8):
        if prompt is not None and prompt != self.target_text_prompt:
            self._set_text_target(prompt)

        x = torch.sigmoid(logits)              # map to [0,1]
        # add a little blur
        if x.shape[-2:] >= (32, 32):
            k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=x.dtype, device=x.device)
            k = (k / k.sum()).view(1,1,3,3).repeat(1,3,1,1)
            x = F.conv2d(x.repeat(1,3,1,1), k, padding=1) / 1.0  # mild blur
            
        x = self.preprocess(x.to(self.device))    # [B,3,224,224]

        B = x.shape[0]
        sims = []
        for _ in range(K):
            # random jitter + mild random-resize crop
            xx = x
            if jitter > 0:
                pad = jitter
                xx = F.pad(xx, (pad,pad,pad,pad), mode="reflect")
                offx = torch.randint(0, 2*pad+1, (1,), device=xx.device).item()
                offy = torch.randint(0, 2*pad+1, (1,), device=xx.device).item()
                xx = xx[..., offy:offy+224, offx:offx+224]
            if scale is not None:
                s = float(torch.empty(1).uniform_(scale[0], scale[1]))
                S = int(round(224 * s))
                xx = F.interpolate(xx, size=(S,S), mode="bilinear", align_corners=False)
                xx = F.interpolate(xx, size=(224,224), mode="bilinear", align_corners=False)

            with torch.autocast(device_type=self.device.type, enabled=(self.device.type=='cuda')):
                feat = self.clip_model.encode_image(xx).float()
            feat = F.normalize(feat, dim=-1)
            sims.append((feat * self.target_embed).sum(dim=-1))  # [B]

        sim = torch.stack(sims, dim=0).mean(0)  # [B]

        # move away from "negative" objectives (i.e. static etc.)
        if getattr(self, "_neg", None) is not None:
            neg = []
            for xx in [x]:  # or your cutouts
                with torch.autocast(device_type=self.device.type, enabled=(self.device.type=='cuda')):
                    f = self.clip_model.encode_image(xx).float()
                f = F.normalize(f, dim=-1)
                neg.append((f * self._neg).sum(dim=-1))
            neg_sim = torch.stack(neg).mean(0)
            sim = sim - 0.15 * neg_sim  # α≈0.1–0.2
        loss = 1.0 - sim
        return loss.to(logits.dtype)

    def forward(self, design: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(design.to(self.device))
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type=='cuda')):
            img_feat = self.clip_model.encode_image(x).float()
        img_feat = F.normalize(img_feat, dim=-1)
        sim = (img_feat * self.target_embed).sum(dim=-1)  # [B]
        loss = self.weight * (1.0 - sim).mean()
        return loss.to(design.dtype)
