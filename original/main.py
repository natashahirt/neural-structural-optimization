"""
A script to generate a video of the FEA and CLIP losses for a given prompt.
"""

import torch
import argparse


from CLIP_utils import load_clip_model, GenerateCrops, Prompts
from PIL import Image
from loss_utils import LossObject
import matplotlib.pyplot as plt
import numpy as np
from train import Optimizer
from torchvision import transforms
import os
import glob

from neural_structural_optimization.problems import get_problem_by_name
from neural_structural_optimization import models, topo_api
from models import MMSD

# Use GPU if available for pytorch functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################################################
# 0. Define Hyperparameters
###################################################################################
s = 0
width = 128//2**s
height = 256//2**s
scale = 4 * 2**s

img_width = width*scale
img_height = height*scale
resizes = 2
multiple = 2**resizes
floors = 4
interval = (height)//floors

model_type = "Ada"# types: CNN, Pix, Ada, Up

if model_type == "Ada":
    width = width // multiple
    height = height // multiple
    interval = interval//multiple


params = {
    'model_type': model_type, # types: CNN, Pix, Ada, Up
    'lr': 2e-1,
    'max_iterations': 200,
    'clip_weight': 1,
    'clip_alpha': 10,
    'compliance_weight': 1,
    'compliance_weight': 1,
    'clip_weight_rn': 10000,

    'clip_rn_alpha': 0,
    
    'plot_frequency': 5,
    'save_frequency': 20,
    'video_frame_frequency': 2,
    'num_augs': 32,
    'affine_augmentation':True,
    'use_arcsin_transform': True,
    'img_width': img_width,
    'img_height': img_height,
    'seed': 12, # None for random generation
    'loss_types': ['compliance_loss', 'clip_loss'], #'compliance_loss', 'clip_loss', 'geometric_loss'
    'test_folder_name': "24_0125_clip_img_outerLines",
    'initial_image_path': "resources/input_images/dSketches/thick_outer_lins.png",
    'invert_image': True,
    'device': device.type,
    'clip_conv_loss_type': 'L2',
    'clip_conv_layer_weights': [1.0, 0.0, 0.0, 0.0, 0.0], # [1.0,0.0,0.0,0.0,0.0] gives cleaner results
    'augment_target_image': True,
    'make_video': True,
    'clip_model_name': 'ViT-B/32', #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    
    'long_description': "", # Clip_and_fea_10_then_fea_10_for_200_itterations

    # SIMP Params
    'penal': 3.0,
    'filter_width': 2.0,

    # Structural Problem Params
    'problem_name': "multistory_building",
    'description': "",
    'width': width,
    'height': height,
    'density': 0.3,
    'mask': None,
    'interval': interval,

    # Update model size
    'resize_num': resizes, # Num of times to resize model, only for adaptive models
    'resize_scale': 2,
    'resize_threshold': 5e-1,
    'convergence_threshold': .05,
    'max_resize_iteration': 50
    }


# Text prompts
all_text_prompts = [
    "Grid", "Lattice", "Skeleton", "Structural", "Cellualar", "Spiral", "Barrel", "Truss", "Branchy",
    "Linear", "Crystalline", "Voronoi", "Lattice", "Circular", "Wavy", "Elliptical", "Rectangular", "Boxy",
    "Simple", "Complex", "Strong", "Weak", "Ordered", "Chaotic", "Beautiful", "Ugly",
    "Architectural", "Building", "Monumental","Tree", "Penrose", "Tiling", "Gradient",
    "Dog", "Cat", "Coral", "Camel", "Human", "Wings", 
    "Nothing",
    "Wood", "Concrete", "Steel",
    "Chair", "Table",
    ]

all_text_phrases = [
    "Triangular", "Triangular_mesh", "square", "squares", "square_mesh", "square_texture", "circular_mesh", "circular_texture",
    "chaotic_circular_mesh", "ordered_circular_mesh", #TEST WITHOUT RANDOMNESS
    "baroque", "baroque_style", "baroque_style_texture", "baroque_mesh", "baroque_building",
    "simple_thick_lines", "simple_thin_lines",  "web", "simple_thin_web", "simple_thick_web", "simple_web", "web_structure", "simple_web_structure",
    "branch_coral", "coral_branchy", "branchy", "coral", "branchy_cactus", "branchy_spiky", "spiky", "cactus"
    "truss", "truss_structure", "crystal", "crystal_lattice", "crystal_structure", "crystal_lattice_structure",#TEST WITHOUT RANDOMNESS
    "ordered_structure", "chaotic_structure", "thick_structure", "thin_structure", "strong_structure", "weak_structure", "elegant_smooth_structure",
    "messy_curvy_structure", "square_ordered_structure", "triangular_structure",
    "curvy", "curvy_mesh", "spiral", "spiral_curves", "spiral_mesh", "wavy", "wavy_structure", "zigzag", "stacked_zigzags", "zigzag_structure",
    "Wassily_Kandinsky", "Zaha_Hadid", "Frank_Gehry", "Leonardo_Davinci_sketches", "penrose","wassily_kandinsky_structure","wassily_kandinsky_building",
    "Linear_grid_mesh", "linear_grid", "linear_grid_structure", #TEST WITHOUT RANDOMNESS
    "Skeleton_bones_with_skull_at_center", "structural_skeleton", "skeleton", "human_skeleton", "building_skeleton", "skeletal_bones", "skeleton_pelvis",
    "skeleton_lungs"
    "voronoi", "voronoi_mesh",
    "tiling", "square_tiling", "voronoi_tiling", "traingular_tiling", "penrose_tiling", "spiral_tiling", "ordered_tiling", "chaotic_tiling",
    "2d_tiling", "2d_square_tiling", "wassily_kandinsky_tiling",
    "tree", "coral_tree", "branchy_tree", "tree_branches"
    "one_large_circle_in_the_center",
]

all_text_phrases = [
    "triangle", "square", "circle", "pentagon",
    "Baroque_Architecture", "Romanesque_Architecture", "Art_Nouveau", "Art_Deco", "Brutalist_Architecture", "Classical_Architecture", "Greek_Architecture",
    "Baroque", "Romanesque", "Art_Nouveau", "Art_Deco", "Brutalist", "Classical", "Greek",
    "coral", "coral_large", "coral_texture", "coral_wavy", "coral_branchy", "branchy_coral", "coral_spiky", "branchy_spiky", "tree_spiky",
    "tiling_square", "tiling_triangular", "triangular_tiling", "square_tiling",
    "structure_smooth", "structure_smooth_with_a_fluid_form", "structure_smooth_with_a_solid_form", "structure_smooth_with_a_curvy_form", "structure_smooth_with_a_linear_rectangular_grid_form",
    "crystal_lattice_structure", "crystal_lattice", "spiky_structure", 
    "plaid_pattern", "plaid_texture", "plaid_pattern_structure", "plaid_texture_structure", "plaid", "organic_pattern", "organic_wavy_pattern", "organic_curvy_pattern", "coral_pattern", 
    "organic_coral_pattern", "geometric_patterns", "stripes", "geometric", "polka_dots", "floral_pattern", "paisley_pattern", "chevron", "damask", "diamonds", "argyle", "mosaic", "strucutral, _mosaic", "organic_mosaic",
    "fish_scale", "smooth_structure", "wavy_structure", "scalloped_structure", "grainy", "fuzzy_structure", "crinkled_structure", "woven_structure", "bumpy_structure", "embroidered_structure",
    "linear_rectangular_grid", "porous_structure", "a_flat_surface_with_holes", "skeletal_bones", "skeleton", "skeletal_structure", "human_skeleton", "stacked_plaids", "layered_stripes",
    "preforated_design", "preforated_structure", "foam_structure", "gauze_fabric", "fabric", "grid_pattern", "waffle_weave", "coral_structure", "sponge_texture","leaf_vain_patterns", "moire_pattern", "holes",
    "A_large_structure_with_holes_and_a_strong_column_with_litte_branches", "voronoi", "elegant_vornoi", "elegant_structure_with_voronoi", "spiral_structure", "voronoi_mesh"    
]

# chosen_text_phrases = ["Picasso_painting", "surrealism_melting", "cubism", "elegant_voronoi_structure", 
#                        "islamic_geometric_pattern" , ""]
# chosen_text_phrases = ["Picasso_painting", "Wassily_Kandinsky", "Matisse_cutout", "Memphis_design", "surrealist", "renee_magritte_surrealism", 
#                        "islamic_geometric_pattern", 
#                        "Frida_Kahlo_Painting", 
#                        "linear_rectangular_grid", "human_skeleton_bones", "foam_structure", "weaving_fabric_pattern", "fuzzy_structure", 
#                        "fish_scale", "teeth", "grid_pattern", "coral", "A_large_structure_with_holes_and_a_strong_column_with_litte_branches",
#                        ]

# chosen_text_phrases = ["Cubism", "Picasso_painting", "Wassily_Kandinsky", "Coral",
#                        "elegant_voronoi_structure", "islamic_geometric_pattern",
#                        "Arabesque_calligraphy", "Arabesque_geometric_art", "Arabesque_geometric_pattern",
#                        "Arabesque_line_drawing", "Greek_line_drawing", "Japanese_line_drawing",
#                        "Memphis_design_line_drawing", "Arabesque_mashrabiya", "Arabesque_mashrabiya_designed_with_arabesque_calligraphy",
#                        "Human_skeleton_bones", "Greek_architecture", "surrealism_melting", "melting_wax",
#                        "Frida_Kahlo", "Henri_Mattise_shapes", "Matisse_shape_cutouts", "Henri_Matisse_line_drawing",
#                        "Memphis_design"
# ]

chosen_text_phrases = ['Frida_Kahlo', 'Frida_Kahlo_painting', 'Frida_Kahlo_drawing', 'Frida_Kahlo_2d_vector', 'Frida_Kahlo_2d_drawing']

# chosen_text_phrases = ["Arabesque_calligraphy", "Arabesque_geometric_pattern", 
#                        "Arabesque_mashrabiya_designed_with_arabesque_calligraphy", "coral", 
#                        "cubism", "cubism_line_drawing", "elegant_voronoi_structure", "Henri_Mattise_line_drawing", 
#                        "Henri_Mattise_shapes", "Human_skeleton_bones", "islamic_geometric_pattern", "melting_wax",
#                        "surrealism_melting", "Wassily_Kandinsky", 'Picasso_painting']

chosen_text_phrases = ["Arabesque_calligraphy", 
                       "coral", 
                       "Henri_Mattise_line_drawing", 
                       "Henri_Mattise_shapes",
                       "Human_skeleton_bones", 
                       "melting_wax", 
                       "Wassily_Kandinsky", 
                       'Picasso_painting']

# chosen_text_phrases = ['Bauhaus', 'Bauhaus_art', 'Bauhaus_art_movement', 'Bauhaus_art_style', 'Bauhaus_style', 'Bauhaus_movement',
#                        'Bauhaus_line_art', 'Bauhaus_line', 'Bauhaus_shapes', 'Bauhaus_graphics', 'Bauhaus_design', 'Bauhaus_graphic_design', 
#                        'Bauhaus_cutouts', 'Bauhaus_shape_cutouts', 'Bauhaus_design_movement', 'Bauhaus_pattern', 'Bauhaus_graphic_pattern',
#                        'Bauhaus_texture', 'Bauhaus_style_pattern', 'Bauhaus_art_pattern', 'Bauhaus_geometry', 'Bauhaus_style_geometry', 'Bauhaus_design_geometry',
#                        'Bauhaus_2d_vector', 'Bauhaus_image', 'Bauhaus_illustration', 'Bauhaus_flat_illustration', 'Bauhaus_drawing']

# chosen_text_phrases = ["Da_Vanvi_sketches", "Frida_Kahlo_painting", "spiky_structure", "elegant_voronoi_structure", "Square", "Triangular_mesh", "Organic_spiraling_forms", "Fluid_organic_forms", "Crystal_lattice_structure", "web", "simple_thick_lines"]

# Define the text prompts
text_prompts = Prompts()
# text_prompts.add_prompt('linear', 1.0)
# text_prompts.add_prompt('na', 1.0)
text_prompts.add_prompt('square', 1.0)
# text_prompts.add_prompt('Skeleton', 1.0)

# text_prompts.add_prompt('shadow', 1.0)



# text_prompts.add_prompt('a cool building that is structural large and linearly rectangular with five floors ', 1.0)



###################################################################################
# 1. Define the model
###################################################################################


# def init_weights_mask(model):
#     if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.Linear):
#         model.weight[:30] = -20

# # # Apply the initial variables
# # model.apply(init_weights_mask)


###################################################################################
# 2. Run the model
###################################################################################
# Clean memory



## Testing different parameters for geometric loss
# from itertools import product

# # Define the array
# array = [0.0, 0.0, 0.0, 0.0, 0.0]

# # Generate all possible combinations of 0.0 and 1.0 for the array
# combinations = list(product([0.0, 1.0], repeat=len(array)))

# # Display all combinations

# combo = combinations[26]

# params['clip_conv_layer_weights'] = list(combo)
# description = ""
# for c in combo:
#     description += f"{int(c)}"

# params['description'] = "no_geo_loss"

# folder = "resources/input_images/dSketches"
# initial_image_paths = glob.glob(os.path.join(folder, '*'))


# for initial_image_path in initial_image_paths:
# for cw in clip_weights:

# for text in all_text_phrases:

# for resize in resizes:
# alphas = [0.5, 1.5,  2, 5, 50, 500]
# for alpha in alphas:
# for i in range(1):
for text in chosen_text_phrases:
    try:
        
        text_prompts = Prompts()
        text_prompts.add_prompt(text, 1.0)

        
        # Reset parameters
        params['width'] = width
        params['height'] = height
        params['interval'] = interval
        

        # params['clip_weight'] = cw
        # params['initial_image_path'] = initial_image_path

        # Initiate model and image
        mmsd = MMSD(params, directory=params['test_folder_name'], device=device)
        mmsd.set_seed(params['seed'])
        # hh, ww = mmsd.structural_model.z.shape[1:3]       
        # init_tensor = torch.ones(1, hh, ww) * params['density']
        # init_tensor[:,:,width//2-5:width//2+5] = 1.
        # mmsd.structural_model.z = torch.nn.Parameter(init_tensor, requires_grad=True)


        # mmsd.init_weight_random()

        mmsd.init_weight_with_image(params['initial_image_path'], invert_image=params['invert_image'])

        # params['loss_types'] = ['compliance_loss', 'clip_loss']
        # params['max_iterations'] = 1000
        # mmsd.optimize(params, text_prompts=text_prompts, target_image_path=params['initial_image_path'], seed=params['seed'])

        # params['loss_types'] = ['clip_loss']
        # params['max_iterations'] = 21
        # mmsd.optimize(params, text_prompts=text_prompts, target_image_path=params['initial_image_path'], seed=params['seed'])

        
        # params['loss_types'] = ['compliance_loss', 'clip_loss', 'geometric_loss']
        

        params['loss_types'] = ['compliance_loss', 'clip_loss'] 
        # params['loss_types'] = ['compliance_loss']
        params['max_iterations'] = 200
        mmsd.optimize(params, text_prompts=text_prompts, target_image_path=params['initial_image_path'], seed=params['seed'])

            # params['loss_types'] = ['compliance_loss']
            # params['max_iterations'] = 10
            # mmsd.optimize(params, text_prompts=text_prompts, target_image_path=params['initial_image_path'], seed=params['seed'])

        # Plot the losses
        mmsd.plot_and_save_final(params['img_width'], step_factor=1, params=params, log=True)

        # Generate video
        mmsd.make_video(fps=15)
    except Exception as e:
        print(e)