"""
Contains a list of functions for CLIP and image preprocessing and evaluation
"""

import clip
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia.augmentation as K
from torch.nn import functional as F
from dataclasses import dataclass


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



##########################################
# 1.0 | CLIP FUNCTIONS
##########################################
def load_clip_model(clip_model_name):
  "A function to load clip with right device"  
  clip_model, _transforms = clip.load(clip_model_name, jit=False)
  # By saying the clip model doesn't require gradients, we are essentially freezing its layers
  # Thus, backgpropagation won't optimize CLIP's weights
  return clip_model.eval().requires_grad_(False).to(device), _transforms



def preprocess_image_for_clip(x):
  "Normalizes image to match CLIP datast distribution"
  normalize = transforms.Normalize(
      mean=[0.48145466, 0.4578275, 0.40821073],
      std=[0.26862954, 0.26130258, 0.27577711])
  return normalize(x)



def encode_text(text, clip_model):
  tokenized_text = clip.tokenize(text).to(device)
  return clip_model.encode_text(tokenized_text).float()


@dataclass
class Prompt:
    text: str
    weight: float

    def __str__(self):
        """Returns a string representation of the prompt when map(str, Prompt) is called"""
        return self.text


def create_prompt_class(prompts, custom_weights=None):
  """
  A function to create a Prompt class for CLIP.
  """
  prompt_list = []

  # If no custom weights are provided, give all prompts the same weight
  if custom_weights is None:      
    for prompt in prompts:
        prompt_list.append(Prompt(text=prompt, weight=1.0))

  else:
     for i, prompt in enumerate(prompts):
        prompt_list.append(Prompt(text=prompt, weight=custom_weights[i]))

  return prompt_list       

class Prompts:
    """A class to manage a collection of Prompt instances.

    This class provides methods to add new prompts and to retrieve the
    index of a prompt based on its text or weight.

    Attributes:
        prompts (List[Prompt]): A list to store Prompt instances.
    """
    def __init__(self):
        """Initializes the Prompts with an empty list."""
        self.prompts = []

    def add_prompt(self, text: str, weight: float=1.0):
        """Adds a new prompt to the collection.

        Args:
            text (str): The text of the prompt.
            weight (float): The weight of the prompt.
        """
        self.prompts.append(Prompt(text, weight))

    def find_index_by_text(self, text: str) -> int:
        """Finds the index of a prompt by its text.

        Args:
            text (str): The text to search for.

        Returns:
            int: The index of the prompt with the given text, or -1 if not found.
        """
        for index, prompt in enumerate(self.prompts):
            if prompt.text == text:
                return index
        return -1  # Return -1 if not found

    def find_index_by_weight(self, weight: float) -> int:
        """Finds the index of a prompt by its weight.

        Args:
            weight (float): The weight to search for.

        Returns:
            int: The index of the prompt with the given weight, or -1 if not found.
        """
        for index, prompt in enumerate(self.prompts):
            if prompt.weight == weight:
                return index
        return -1  # Return -1 if not found

    def __iter__(self):
        """Allows iteration over the collection of Prompt instances."""
        return iter(self.prompts)
    
    def __repr__(self):
        combined_prompt = '_'.join(map(str, self.prompts))
        return combined_prompt
         




##########################################
# 2.0 | Image preprocessing functions
##########################################

class GenerateCrops(torch.nn.Module):
  def __init__(self, crop_size, batch_size, min_original_size=480, noise=0.1):
      """
      A module for generating various image augmentations. It applies a series of transformations 
      to an input image to create augmented versions, which can be used for data augmentation in 
      image processing tasks.

      Args:
          crop_size (int): The desired size for output image crops.
          batch_size (int): The number of images in one batch.
          min_original_size (int, optional): The minimum size of the original images before cropping. 
                                            Defaults to 480.
          noise (float, optional): A parameter to control the addition of random noise. Defaults to 0.1.

      Methods:
          forward(input): Processes the input image and applies the augmentations.
          visualize_augmentations(batch): Visualizes the augmented images if requested.
      """
      super().__init__()

      self.batch_size = batch_size

      scale_min = crop_size/min_original_size

      self.augs = torch.nn.Sequential(
          K.RandomResizedCrop(size=(crop_size, crop_size), scale=(scale_min, 1.0), cropping_mode ="resample"),
          K.RandomHorizontalFlip(p=0.5),
          K.RandomSharpness(0.3,p=0.4),
          K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
          K.RandomPerspective(0.2,p=0.4),
          K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
          )
      self.noise = noise
 
 
  def forward(self, input):
    """
    When an input image (or a batch of images) is passed to this module, 
    this method is called to perform the augmentations.
    """
    # Create a batch of identical images
    cutouts = torch.cat(self.batch_size*[input],dim=0)
    #  initializes several instance variables and creates an augmentation pipeline for the batch
    batch =  self.augs(cutouts)
    self.batch = batch


    if self.noise:
        facs = batch.new_empty([self.batch_size, 1, 1, 1]).uniform_(0, self.noise)
        batch = batch + facs * torch.randn_like(batch)
    return batch
  
  
  def visualize_augmentations(self, batch):
    "Visualizes the augmentations given the image batch"
    fig, axs = plt.subplots(1, self.batch_size, figsize=(self.batch_size * 3, 3))
    for i, img in enumerate(batch):
        img = img.permute(1, 2, 0)  # Change the channel order for visualization
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] for displaying
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()
    

##########################################
# 3.0 | Evaluation functions
##########################################
