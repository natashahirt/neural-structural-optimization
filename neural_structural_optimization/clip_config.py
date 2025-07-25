"""
Configuration utils for clip integration
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class CLIPConfig:
    """config for clip loss integration"""

    device: str = 'cpu'
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

    return {
        'model_config': {
            'device': config.device,
            'model_name': config.model_name
        },
        'weight': config.weight,
        'target_text_prompt': config.target_text_prompt,
        'target_image_path': config.target_image_path
    }
