import torch
import clip
import numpy as np
import tensorflow as tf
from PIL import image
import io

class CLIPLoss:
    
    def __init__(self, device='cpu', model_name='ViT-B/32'):
        # init CLIP model and tokenizer
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def design_to_image(self, design, size=(224,224)):
        # ensure 2d
        if design.shape[0] == 1:
            design = design[0]

        # convert to numpy
        design_np = design.numpy() # convert to numpy array
        design_np = (design_np - design_np.min()) / (design_np.max() - design_np.min()) * 255 # normalize to range 0-255
        design_np = design_np.astype(np.uint8)

        # convert to PIL image
        image = Image.fromarray(design_np, mode='L') # greyscale
        image = Image.resize(size, Image.LANCZOS)

        # convert to rgb for CLIP
        rgb_image = Image.new('RGB', size)
        rgb_image.paste(image, (0,0))

        return rgb_image
    
    def get_text_loss(self, design, text_prompt, temp=1.0):
        """
        design: tensorflow tensor of structural design
        text_prompt: text description of desired image
        temp: temp for clip similarity comparison

        returns tensorflow tensor representing clip loss
        """

        design_image = self.design_to_image(design)

        # preprocess for clip
        design_input = self.preprocess(design_image).unsqueeze(0).to(self.device)
        target_input = clip.tokenize([text_prompt]).to(self.device)

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
        return tf.constant(loss, dtype=tf.float32)

    def get_image_loss(self, design, image_path, temp=1.0):
        """
        design: tensorflow tensor of structural design
        image_path: path to the reference image
        temp: temp for clip similarity comparison

        returns tensorflow tensor representing clip loss
        """

        target_image = Image.open(image_path).convert('RGB')
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