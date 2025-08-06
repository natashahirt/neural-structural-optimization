"""
Configuration utils for clip integration
"""

from dataclasses import dataclass
from typing import Optional
import tensorflow as tf
from transformers import TFCLIPModel, CLIPTokenizer

@dataclass
class CLIPConfig:
    """config for clip loss integration"""

    device: str = 'auto'
    model_name: str = 'openai/clip-vit-base-patch32'

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
        if self.device == 'auto':
            if tf.config.list_physical_devices('GPU'):
                print("GPU available, using GPU for CLIP")
                return 'gpu'
            else:
                print("GPU not available, using CPU for CLIP")
                return 'cpu'
        elif self.device == 'gpu':
            if tf.config.list_physical_devices('GPU'):
                print("Using GPU for CLIP")
                return 'gpu'
            else:
                print("GPU requested but not available, falling back to CPU")
                return 'cpu'
        elif self.device == 'cpu':
            print("Using CPU for CLIP")
            return 'cpu'
        else:
            print(f"Unknown device '{self.device}', falling back to CPU")
            return 'cpu'

    def check_clip_availability(self):
        try:
            import clip
            from PIL import Image
            # basic functionality
            model = TFCLIPModel.from_pretrained(self.model_name)
            tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            # tokenization
            _ = tokenizer(["test"], return_tensors="tf")
            print("CLIP (TF) and tokenizer loaded successfully.")
            return (True, f"CLIP (TF) available on '{self.device}'")
            # image preprocessing
            dummy_image = Image.new('RGB', (224,224), color='white')
            image_input = preprocess(dummy_image).unsqueeze(0).to(self.device)

            return (True, f"CLIP (TF) available on '{self.device}'")
        except ImportError as e:
            return (False, f"HuggingFace Transformers not installed: {e}")
        except Exception as e:
            return (False, f"CLIP (TF) error: {e}")

def create_clip_config(
    target_text_prompt=None,
    target_image_path=None,
    weight=0.1,
    device='cpu',
    model_name='openai/clip-vit-base-patch32'
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
        print("CLIP (TF) loss will be disabled.")
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
