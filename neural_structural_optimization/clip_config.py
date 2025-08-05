"""
Configuration utils for clip integration
"""

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class CLIPConfig:
    """config for clip loss integration"""

    device: str = 'auto'
    model_name: str = 'ViT-B/32'

    # loss weighting
    weight: float = 0.1

    # target spec (text/image)
    target_text_prompt: Optional[str] = None
    target_image_path: Optional[str] = None

    def __post_init__(self):
        """validate config"""
        if self.target_text_prompt is None and self.target_image_path is None:
            raise ValueError("Must specify either target_text_prompt or target_image_path.")

        if self.target_text_prompt is not None and self.target_image_path is not None:
            raise ValueError("Cannot specify both target_text_prompt and target_image_prompt.")

        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        if self.device == 'auto' or self.device == 'cuda':
            if torch.cuda.is_available():
                print("CUDA available, using CUDA for CLIP")
                return 'cuda'
            else:
                print("CUDA not available, using CPU for CLIP")
                return 'cpu'
        elif self.device == 'cpu':
            print("Using CPU for CLIP")
            return 'cpu'
        else:
            print(f"Requested device '{self.device}' unknown. Falling back to CPU.")
            return 'cpu'

    def check_clip_availability(self):
        try:
            import clip
            from PIL import Image
            # basic functionality
            model, preprocess = clip.load(self.model_name, device=self.device)
            # tokenization
            text = clip.tokenize(["test"]).to(self.device)
            # image preprocessing
            dummy_image = Image.new('RGB', (224,224), color='white')
            image_input = preprocess(dummy_image).unsqueeze(0).to(self.device)
            # encoding
            with torch.no_grad():
                model.encode_text(text)
                model.encode_image(image_input)
            return (True, f"CLIP available on '{self.device}'")
        except ImportError as e:
            return (False, f"CLIP not installed: {e}")
        except Exception as e:
            return (False, f"CLIP error: {e}")

def create_clip_config(
    target_text_prompt=None,
    target_image_path=None,
    weight=0.1,
    device='cpu',
    model_name='ViT-B/32'
):
    config = CLIPConfig(
        target_text_prompt=target_text_prompt,
        target_image_path=target_image_path,
        weight=weight,
        device=device,
        model_name=model_name
    )

    clip_available, status_message = config.check_clip_availability()
    if not clip_available:
        print(status_message)
        print("CLIP loss will be disabled.")
        config.weight = 0.0
    else:
        print(status_message)

    return {
        'model_config': {
            'device': config.device,
            'model_name': config.model_name
        },
        'weight': config.weight,
        'target_text_prompt': config.target_text_prompt,
        'target_image_path': config.target_image_path
    }
