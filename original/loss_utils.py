"""
A script to store the different evaluation functions
"""

import torch
from CLIP_utils import encode_text, preprocess_image_for_clip, load_clip_model, GenerateCrops
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms




def evaluate_structural_compliance(img, model):
    """
    An FEA method implemented in the model to evaluate the compliance loss.
    Compliance is a measure of how much a structure deforms under a given load.
    A more compliant structure is less stigg and deforms more under the same load.
    """
    loss = model.loss()  
    return loss(img, model.env)



def clip_weight_fn(loss_decay, min_weight, max_weight, half_itter, i):
    "returns the function to calculate clip weight, either a decaying functaion or a min clip weight"
    if loss_decay:
      fn = lambda x : max_weight / 10**((x - half_itter) / (half_itter/np.log10(max_weight/min_weight)))
    else:
      fn = lambda x : min_weight
    clip_weight = int(fn(i))
    return clip_weight


def l2_layers(x_conv_features, y_conv_features):
    """
    Summing the L2 distance loss from the four layers of clip (x1, x2, x3, x4)
    """
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(x_conv_features, y_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPLoss(torch.nn.Module):
    def __init__(self, clip_model, clip_model_rn, clip_preprocess, clip_preprocess_rn, params):
        """
        change params to args
        add description
        """
        super(CLIPLoss, self).__init__()

        # Set clip model in class
        self.clip_model_rn = clip_model_rn
        self.clip_model = clip_model

        # Set loss types and validate
        self.clip_conv_loss_type = params['clip_conv_loss_type']
        self.augment_target_image = params['augment_target_image']
        self.clip_conv_layer_weights = params['clip_conv_layer_weights']
        # assert self.clip_conv_loss_type in ["L2", "Cos", "L1"]

        # Map of distance metrics for loss calculation
        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                # "Cos": cos_layers
            }
        
        # Get model layers
        layers = list(self.clip_model_rn.visual.children())
        init_layers = torch.nn.Sequential(*layers)[:8]
        # We only want the first 4 layers for the geometric loss
        self.layer1 = layers[8]
        self.layer2 = layers[9]
        self.layer3 = layers[10]
        self.layer4 = layers[11]
        self.att_pool2d = layers[12]
        # self.layer1 = layers[-5]
        # self.layer2 = layers[-4]
        # self.layer3 = layers[-3]
        # self.layer4 = layers[-2]
        # self.att_pool2d = layers[-1]

        # Get transformations here
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.normalize_transform_rn = transforms.Compose([
            clip_preprocess_rn.transforms[0],  # Resize to 224
            clip_preprocess_rn.transforms[1],  # CenterCrop to 224 by 224
            clip_preprocess_rn.transforms[-1],  # Normalize
        ])

        # # Get affine transformations if defined
        augemntations = []
        if params['affine_augmentation']:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        # self.device = params['device']
        # self.args = args
        self.num_augs = params['num_augs']
        self.params = params
        self.device = params['device']
        self.use_arcsin_transform = params['use_arcsin_transform']


    def forward(self, image, target, params=None):
        """
        Parameters
        ----------
        image: Torch Tensor [N, C, H, W]
        target: Torch Tensor [1, C, H, W] representing a target image 
            or string representing the text description
        """
        # Define hyperparameters if new ones are input
        if params:
            self.use_arcsin_transform = params['use_arcsin_transform']
            self.num_augs = params['num_augs']


        if type(target) == str:
            loss = self.evaluate_image_semantics(image, target, self.use_arcsin_transform)
        else:
            loss = self.evaluate_image_to_image(image, target)
        return loss


    def forward_inspection_clip_resnet(self, x):
        """
        Perform a forward pass through the initial part of the visual part of CLIP (conv1, conv2, conv3)
        and the 4 Sequential layers/blocks. Get the target image with a final pass over
        the maxpool layer of clip, however, the loss we need is from the 4th layer only.
        Note that this function is not general, it is specefic to the RESNET101 CLIP architecture.
        **Maybe expand it to include the VIT architecture as it seemed to do better with sketch thickness?

        
        Args:

        Returns:

        """

        def stem(m, x):
            for conv, bn, relu in [(m.conv1, m.bn1, m.relu1), (m.conv2, m.bn2, m.relu2), (m.conv3, m.bn3, m.relu3)]:
                x = relu(bn(conv(x)))  # Apply conv, batch norm, and ReLU activation

            x = m.avgpool(x)  # Apply average pooling
            return x

        # Cast input tensor to the same data type as the first convolution layer
        x = x.type(self.clip_model_rn.visual.conv1.weight.dtype)

        # Process input through the stem layers
        x = stem(self.clip_model_rn.visual, x)

        # Sequentially process the tensor through ResNet layers
        x1 = self.clip_model_rn.visual.layer1(x)
        x2 = self.clip_model_rn.visual.layer2(x1)
        x3 = self.clip_model_rn.visual.layer3(x2)
        x4 = self.clip_model_rn.visual.layer4(x3)

        y = self.clip_model_rn.visual.attnpool(x4)

        return y, [x, x1, x2, x3, x4]
    

    def evaluate_image_to_image(self, input_image, target_image):
        # Initialize a dictionary to store convolutional layer losses
        conv_loss_dict = {}

        # Move input image and target images to the specified device
        x = input_image #.to(self.device).unsqueeze(0)
        y = target_image #.to(self.device).unsqueeze(0)

        # Normalize the input and target images
        
        # x_normalized, y_normalized = [self.normalize_transform_rn(x)], [
        #     self.normalize_transform_rn(y)]
        
        # # Apply SAME augmentations to the input images and target images
        # for n in range(self.num_augs):
        #     augmented_pair = self.augment_trans(torch.cat([x, y]))
        #     x_normalized.append(augmented_pair[0].unsqueeze(0))

        #     # Need to augment the target images as well incase input image
        #     # is not square since centercrop crops a square image.
        #     if self.augment_target_image:
        #         y_normalized.append(augmented_pair[1].unsqueeze(0))

        # # Concatenate augmented images and move them to the device
        # x_augmented = torch.cat(x_normalized, dim=0).to(self.device)
        # y_augmented = torch.cat(y_normalized, dim=0).to(self.device)

        generate_crops = GenerateCrops(self.clip_model.visual.input_resolution, self.num_augs, min_original_size=min(480, 480))
        x_augmented = generate_crops(x).to(self.device)
        y_augmented = generate_crops(y).to(self.device)        

        # Compute features using the CLIP model
        x_fc_features, x_conv_features = self.forward_inspection_clip_resnet(
            x_augmented)
        y_fc_features, y_conv_features = self.forward_inspection_clip_resnet(
            y_augmented.detach())
                
        # Calculate convolutional layer losses
        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            x_conv_features, y_conv_features)
        
        conv_loss_total = 0
        for layer, w in enumerate(self.clip_conv_layer_weights):
            if w:
                # conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w
                conv_loss_total += conv_loss[layer] * w

        # This part would be calculating fully connected clip layer
        # # Calculate fully connected layer loss, if applicable
        # if self.clip_fc_loss_weight:
        #     # fc distance is always cos
        #     fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
        #                ys_fc_features, dim=1)).mean()
        #     conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        return conv_loss_total

                                
    def evaluate_image_semantics(self, image, text: str, use_arcsin_transform: bool = True):
        """
        Computes a loss value indicating the similarity between an input image and a text description. 
        Encodes both text and image crops, calculates distances between these encodings, and optionally 
        applies an arcsin transformation to the distances.

        Args:
            image: The input image for evaluation.
            text: Text description to compare against the image.
            clip_model: The loaded CLIP model.
            n_crops: Number of image crops for analysis (default: 64).
            use_arcsin_transform: Flag to apply arcsin transformation to distances (default: True).

        Returns:
            A float representing the similarity loss value between image crops and text.
        """
        # # Augment images
        # augmented_images = [self.normalize_transform(image)]
        # for n in range(self.num_augs):
        #     augmented_image = self.augment_trans(image)
        #     augmented_images.append(augmented_image)
        # # concatenate augmented images into a tensor
        # augmented_images = torch.cat(augmented_images, dim=0).to(self.device)

        # Generate crops works better than the code above, explore why...
        generate_crops = GenerateCrops(self.clip_model.visual.input_resolution, self.num_augs, min_original_size=min(480, 480))
        augmented_images = generate_crops(image)
        
        # Encode the text into a vector representation using CLIP
        encoded_text = encode_text(text, self.clip_model)
        # Preprocess the image crops for compatibility with the CLIP model.
        normalized_images = preprocess_image_for_clip(augmented_images)
        # Encode the image crops into a vector representation using CLIP
        crops_encodings = self.clip_model.encode_image(normalized_images)

        

        #     # Need to augment the target images as well incase input image
        #     # is not square since centercrop crops a square image.
        #     if self.augment_target_image:
        #         y_normalized.append(augmented_pair[1].unsqueeze(0))

        # Normalize the encodings of the image crops and text.
        normalized_crops_encodings = F.normalize(crops_encodings, dim=1)
        normalized_encoded_text = F.normalize(encoded_text)

        # Calculate the distance between each normalized image crop encoding and the normalized text encoding.
        encoding_distances = torch.norm(normalized_crops_encodings - normalized_encoded_text ,dim=1)

        # Optionally apply the arcsin transform to the distances.
        if use_arcsin_transform:
            encoding_distances = torch.arcsin(encoding_distances/2)**2

        # Get average similarity loss between images
        loss = torch.mean(encoding_distances)

        return loss




class LossObject(torch.nn.Module):
    def __init__(self, structural_model, clip_model, clip_model_rn, clip_preprocess, clip_preprocess_rn, params):
        """
        """
        super().__init__()

        # initializing losses
        self.compliance_loss = None
        self.clip_loss = None
        self.clip_loss_raw = None
        self.geometric_loss = None
        self.geometric_loss_raw = None
        self.total_loss = None
        
        self.loss_dict = {
            'compliance_loss': [self.compliance_loss],
            'clip_loss': [self.clip_loss],
            'geometric_loss': [self.geometric_loss],
            'clip_loss_raw': [self.clip_loss_raw],
            'geometric_loss_raw': [self.geometric_loss_raw],
            'total_loss': [self.total_loss]
            }

        # defining the model
        self.structural_model = structural_model        
        self.clip_model = clip_model
        self.clip_model_rn = clip_model_rn
        self.clip_loss_object = CLIPLoss(clip_model, clip_model_rn, clip_preprocess, clip_preprocess_rn, params)
        

        # Setting hyperparameters
        self.use_arcsin_transform = params['use_arcsin_transform']
        self.num_augs = params['num_augs']
        self.clip_weight = params['clip_weight']
        self.clip_weight_rn = params['clip_weight_rn']
        self.loss_types = params['loss_types']
        self.use_arcsin_transform = params['use_arcsin_transform']
        self.compliance_weight = params['compliance_weight']


    def forward(self, structural_model_image, clip_image_input, params, prompts=None, target_image=None, loss_combination_fn=None):
        """
        Evaluates the compliance loss and the clip loss if specified in loss_types and outputs the sum
        unless another function is specified.
        0 is compliance loss (fea) and 1 is clip loss (image).

        """
        # Define hyperparameters if new ones are input
        if params:
            self.use_arcsin_transform = params['use_arcsin_transform']
            self.num_augs = params['num_augs']
            self.clip_weight = params['clip_weight']
            self.clip_weight_rn = params['clip_weight_rn']
            self.loss_types = params['loss_types']
            self.use_arcsin_transform = params['use_arcsin_transform']
            self.compliance_weight = params['compliance_weight']


        # Evaluating losses
        for i in range(len(self.loss_types)):

            if self.loss_types[i] == 'compliance_loss':
                self.compliance_loss = evaluate_structural_compliance(structural_model_image.cpu(), self.structural_model)*self.compliance_weight
                self.loss_dict[self.loss_types[i]] = [self.compliance_loss]
                self.clip_weight = self.compliance_loss*params['clip_alpha']
                self.clip_weight_rn = self.compliance_loss*params['clip_rn_alpha']

            elif self.loss_types[i] == 'clip_loss':
                assert prompts is not None, "A text prompt is required to optimize against!"
                clip_losses = [self.clip_loss_object(clip_image_input, prompt.text, params)*prompt.weight for prompt in prompts]
                self.clip_loss_raw = sum(clip_losses)
                self.clip_loss = self.clip_loss_raw*self.clip_weight
                self.loss_dict[self.loss_types[i]] = [self.clip_loss]
                self.loss_dict['clip_loss_raw'] = [self.clip_loss_raw]

            elif self.loss_types[i] == 'geometric_loss':
                assert target_image is not None, "A target image is required to optimize against!"
                self.geometric_loss_raw = self.clip_loss_object(clip_image_input, target_image, params)
                self.geometric_loss = self.clip_loss_object(clip_image_input, target_image, params) * self.clip_weight_rn
                self.loss_dict[self.loss_types[i]] = [self.geometric_loss]
                self.loss_dict['geometric_loss_raw'] = [self.geometric_loss_raw]


        # Update dictionary
        self.loss_dict
        # Define or get the function that combines the different losses 
        if loss_combination_fn is None:
            loss_combination_fn = lambda loss_list: sum(loss_list)

        # Making sure losses that aren't used aren't included in function
        filtered_loss_list = [v[0] for k,v in self.loss_dict.items() if v[0] is not None and 'total_loss' not in k] 
        self.total_loss = loss_combination_fn(filtered_loss_list)
        self.loss_dict['total_loss'] = [self.total_loss]

        return self.total_loss, self.compliance_loss, self.clip_loss, self.clip_loss_raw
    







    