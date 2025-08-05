import torch
import clip
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class CLIPLoss:
    
    def __init__(self, device='cpu', model_name='ViT-B/32', config=None):
        # init CLIP model and tokenizer
        if config is not None:
            if hasattr(config, 'device'):  # CLIPConfig object
                self.device = config.device
                self.model_name = config.model_name
                self.weight = config.weight
                self.target_text_prompt = config.target_text_prompt
                self.target_image_path = config.target_image_path
            else:  # Dictionary
                self.device = config.get('model_config', {}).get('device', 'cpu')
                self.model_name = config.get('model_config', {}).get('model_name', 'ViT-B/32')
                self.weight = config.get('weight', 1.0)
                self.target_text_prompt = config.get('target_text_prompt')
                self.target_image_path = config.get('target_image_path')
        else:
            self.device = device
            self.model_name = model_name
            self.weight = 1.0
            self.target_text_prompt = None
            self.target_image_path = None
            
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        
    def design_to_image(self, design, size=(224,224)):
        # ensure 2d
        if design.shape[0] == 1:
            design = design[0]

        # convert to numpy
        design_np = design.numpy() # convert to numpy array
        # Handle edge case where min == max to avoid division by zero
        design_range = design_np.max() - design_np.min()
        if design_range == 0:
            design_np = np.zeros_like(design_np)
        else:
            design_np = (design_np - design_np.min()) / design_range * 255  # normalize to range 0-255
        design_np = design_np.astype(np.uint8)

        # convert to PIL image
        image = Image.fromarray(design_np, mode='L') # greyscale
        image = image.resize(size, Image.LANCZOS)

        # convert to rgb for CLIP
        rgb_image = Image.new('RGB', size)
        rgb_image.paste(image, (0,0))

        return rgb_image
    
    def get_text_loss(self, design, target_text_prompt, temp=1.0):
        """
        design: tensorflow tensor of structural design
        target_text_prompt: text description of desired image
        temp: temp for clip similarity comparison

        returns tensorflow tensor representing clip loss
        """

        design_image = self.design_to_image(design)

        # preprocess for clip
        design_input = self.preprocess(design_image).unsqueeze(0).to(self.device)
        target_input = clip.tokenize([target_text_prompt]).to(self.device)

        # get clip features
        with torch.no_grad():
            design_features = self.model.encode_image(design_input)
            target_features = self.model.encode_text(target_input)

            # normalize
            design_features = design_features / design_features.norm(dim=-1, keepdim=True)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)

            # similarity
            similarity = (design_features @ target_features.T).squeeze()

        similarity = similarity / temp # higher temp = softer predictions, lower temp = more decisive
        loss = 1.0 - similarity.item() # higher similarity = lower loss
        return tf.constant(loss, dtype=tf.float64)

    def get_image_loss(self, design, target_image_path, temp=1.0):
        """
        design: tensorflow tensor of structural design
        target_image_path: path to the reference image
        temp: temp for clip similarity comparison

        returns tensorflow tensor representing clip loss
        """

        target_image = Image.open(target_image_path).convert('RGB')
        target_image = target_image.resize((224,224), Image.LANCZOS)

        design_image = self.design_to_image(design)

        # preprocess for clip
        design_input = self.preprocess(design_image).unsqueeze(0).to(self.device)
        target_input = self.preprocess(target_image).unsqueeze(0).to(self.device)

        # get clip features
        with torch.no_grad():
            design_features = self.model.encode_image(design_input)
            target_features = self.model.encode_image(target_input)

            # normalize
            design_features = design_features / design_features.norm(dim=-1, keepdim=True)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)

            similarity = (design_features @ target_features.T).squeeze()

        similarity = similarity / temp # higher temp = softer predictions, lower temp = more decisive
        loss = 1.0 - similarity.item() # higher similarity = lower loss
        return tf.constant(loss, dtype=tf.float32)

    def get_batch_text_loss(self, designs, text_prompts, temp=1.0):
        """
        get clip loss for a batch of designs and their text prompts
        """

        losses = [None] * len(designs)
        for i, (design, prompt) in enumerate(zip(designs, text_prompt)):
            loss = self.get_text_loss(design, prompt, temp)
            losses[i] = loss
        return tf.stack(losses)

# def _initialize_clip(self, config: CLIPConfig):

#     self.device = config.device
#     self.clip_model, self.clip_preprocess = clip.load(config.model_name, device=self.device)
#     self.clip_model.eval()

#     if config.target_text_prompt:
#         text = clip.tokenize([config.target_text_prompt]).to(self.device)
#         with torch.no_grad():
#             config.target_clip_features = self.clip_model.encode_text(text).float()
#     elif config.target_image_path:
#         img = Image.open(config.target_image_path).convert("RGB")
#         img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             config.target_clip_features = self.clip_model.encode_image(img).float()

# # Normalize
# config.target_clip_features = config.target_clip_features / config.target_clip_features.norm(dim=-1, keepdim=True)
# config.clip_weight = config.weight
# config.device = config.device

# def compute_clip_loss(self, image_tensor):
#     # Ensure image tensor is the right size for CLIP (224x224)
#     if image_tensor.shape[-2:] != (224, 224):
#         image_tensor = tf.image.resize(image_tensor, (224, 224), method='bilinear') 
#     # Convert to torch tensor and ensure it's on the right device
#     if not isinstance(image_tensor, torch.Tensor):
#         image_tensor = torch.from_numpy(image_tensor.numpy()).to(self.device)
#     # image_tensor: [B, 3, 224, 224] in [-1, 1]
#     image_features = self.clip_model.encode_image(image_tensor)
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     cosine_sim = torch.sum(image_features * self.target_clip_features, dim=-1)
#     return (1.0 - cosine_sim.mean()) * self.clip_weight
