import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import CLIPProcessor, TFCLIPModel
from PIL import Image
from typing import Optional, Union, List, Tuple

class CLIPLoss:
    
    def __init__(self, 
                 model_name: str = 'openai/clip-vit-base-patch32', 
                 config: Optional[object] = None):

        # Extract configuration parameters
        if config is not None:
            # Direct attribute access on config object
            self.model_name = getattr(config, 'model_name', model_name)
            self.weight = getattr(config, 'weight', 1.0)
            self.target_text_prompt = getattr(config, 'target_text_prompt', None)
            self.target_image_path = getattr(config, 'target_image_path', None)
        else:
            # No config provided, use defaults
            self.model_name = model_name
            self.weight = 1.0
            self.target_text_prompt = None
            self.target_image_path = None
            
        # Initialize CLIP model and processor
        self.model = TFCLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # CLIP normalization constants (ViT-B/32)
        self.clip_mean = tf.constant([0.48145466, 0.4578275, 0.40821073])
        self.clip_std = tf.constant([0.26862954, 0.26130258, 0.27577711])

    def preprocess_design(self, design: tf.Tensor, target_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
       
        if len(design.shape) == 2:
            design = tf.expand_dims(design, axis=0)  # [1, H, W]
        if len(design.shape) == 3:
            design = tf.expand_dims(design, axis=-1)  # [1, H, W, 1]
        
        design = tf.cast(design, tf.float32)
        min_val = tf.reduce_min(design, axis=[1, 2, 3], keepdims=True)
        max_val = tf.reduce_max(design, axis=[1, 2, 3], keepdims=True)
        design = (design - min_val) / tf.maximum(max_val - min_val, 1e-6)
        
        design = tf.image.resize(design, size=target_size, method='bicubic')
        design = tf.image.grayscale_to_rgb(design)

        # Normalize to CLIP mean/std
        clip_mean = tf.reshape(self.clip_mean, [1, 1, 1, 3])
        clip_std = tf.reshape(self.clip_std, [1, 1, 1, 3])
        design = (design - clip_mean) / clip_std

        design = tf.transpose(design, perm=[0, 3, 1, 2])

        return design

    def augment_design(self, design: tf.Tensor, num_crops=4, crop_size=224):
        """Simulate Kornia-style batch augmentation for CLIP"""        
        crops = []
        pad_size = crop_size + 20

        for _ in range(num_crops):
            aug = design

            # Manual horizontal flip using control flow
            if tf.random.uniform([]) < 0.5:
                aug = tf.reverse(aug, axis=[1])  # flip horizontally

            # Brightness jitter (safe)
            brightness_delta = tf.random.uniform([], -0.1, 0.1)
            aug = tf.clip_by_value(aug + brightness_delta, 0.0, 1.0)

            # Contrast jitter (manual version)
            factor = tf.random.uniform([], 0.9, 1.1)
            mean = tf.reduce_mean(aug, axis=[0, 1], keepdims=True)
            aug = tf.clip_by_value((aug - mean) * factor + mean, 0.0, 1.0)

            # Resize with padding (center pad, differentiable)
            aug = tf.image.resize_with_crop_or_pad(aug, pad_size, pad_size)

            # Manual random crop (differentiable via slicing)
            max_offset = pad_size - crop_size
            offset_x = tf.random.uniform([], 0, max_offset + 1, dtype=tf.int32)
            offset_y = tf.random.uniform([], 0, max_offset + 1, dtype=tf.int32)
            aug = tf.slice(aug, [offset_y, offset_x, 0], [crop_size, crop_size, 3])

            crops.append(tf.expand_dims(aug, axis=0))  # [1, crop, crop, 3]

        return tf.concat(crops, axis=0)  # [num_crops, crop_size, crop_size, 3]

    def get_text_loss(self, design: tf.Tensor, target_text_prompt: str, temp: float = 1.0, num_crops: int = 4) -> tf.Tensor:
        
        # Preprocess design
        image_input = self.preprocess_design(design)

        # Augment and stack crops
        crops = self.augment_design(design, num_crops=num_crops)  # [N, 224, 224, 3]

        # Normalize to match CLIP stats
        clip_mean = tf.reshape(self.clip_mean, [1, 1, 1, 3])
        clip_std = tf.reshape(self.clip_std, [1, 1, 1, 3])
        crops = (crops - clip_mean) / clip_std
        crops = tf.transpose(crops, [0, 3, 1, 2])  # [N, 3, 224, 224]

        # Run crops through CLIP
        image_features = self.model.get_image_features(pixel_values=crops)  # [N, D]
        image_features = tf.math.l2_normalize(image_features, axis=-1)
        image_embed = tf.reduce_mean(image_features, axis=0, keepdims=True)  # [1, D]

        # Get text embedding
        inputs = self.processor(text=target_text_prompt, return_tensors="tf", padding=True)
        text_embed = self.model.get_text_features(**inputs)
        text_embed = tf.math.l2_normalize(text_embed, axis=-1)

        # Cosine similarity loss
        loss = -tf.keras.losses.cosine_similarity(image_embed, text_embed, axis=-1)
        
        return tf.reduce_mean(loss) * 1e3  # because cosine_similarity is negative

    def get_image_loss(self, design: tf.Tensor, target_image_path: str, temp: float = 1.0) -> tf.Tensor:
        
        raise NotImplementedError("Image-based CLIP loss not yet implemented")

    