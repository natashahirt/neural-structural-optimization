# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import re
import argparse
from PIL import Image
import seaborn
import matplotlib.pyplot as plt
import xarray
import pandas as pd
import numpy as np
import torch

# Enable performance optimizations for modern NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from neural_structural_optimization.structural import utils as pipeline_utils
from neural_structural_optimization.structural import problems
from neural_structural_optimization import models
from neural_structural_optimization.structural import api as topo_api
from neural_structural_optimization.train import ProgressiveTrainer, PixelRefineTrainer, LBFGS_Optimizer
from neural_structural_optimization.structural.problems import StructuralParams
from neural_structural_optimization.models.loss_clip import CLIPLoss

def create_filename_suffix(suffix_str):
    if not suffix_str:
        return ""

    suffix = suffix_str.replace(' ', '_').replace('=', '_').replace(',', '_')
    suffix = suffix.replace('"', '').replace("'", '')
    suffix = suffix.replace('/', '_').replace('\\', '_')
    suffix = suffix.replace(':', '_').replace(';', '_')

    suffix = re.sub(r'[<>:"/\\|?*]', '_', suffix)

    print(f"Using filename suffix: {suffix}")
    
    return f"_{suffix}"

def main():
    """Main function with error handling and progress reporting."""
    print("=" * 60)
    print("Neural Structural Optimization - Multi-Method Comparison")
    print("=" * 60)

    suffix_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    filename_suffix = create_filename_suffix(suffix_str)        
    
    # Ensure output directory exists
    import os
    os.makedirs('script/test_results_pytorch', exist_ok=True)
    
    try:
        # Create problem
        max_iterations = 200

        # Run all optimization methods
        print("\nStarting optimization...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # ViT-B/32, RN50
        clip_loss = CLIPLoss(
            clip_model_name="ViT-B/32", 
            clip_rn_model_name="RN50",
            device=device,
            positive_prompts=["jaguar", "jungle leaves"],   # or "x ray of human skeleton"
            pos_weights=None,                                   # or [1.0, 0.3, ...] matching the prompts
        )

        # note that width and height are targets and not absolute
        params = StructuralParams(
            problem_name = "multistory_building",
            width=50, # 50
            height=100, # 40
            density=0.3,
            num_stories=5,
        )

        params, dynamic_kwargs = pipeline_utils.dynamic_depth_kwargs(params)      

        print("Dynamic kwargs:")
        for key, value in dynamic_kwargs.items():
            print(f"  {key}: {value}")

        print(f"Problem: {params.problem_name}")
        print(f"Dimensions: {params.width}x{params.height}")
        print(f"Max iterations: {max_iterations}")

        # Example 1: PixelModel with L-BFGS optimization 
        # model = models.PixelModel(structural_params=params, clip_loss=clip_loss)
        # trainer = ProgressiveTrainer(model, max_iterations, resize_num=3)
        # ds_history = trainer.train(LBFGS_Optimizer)

        # Example 2: CNNModel with L-BFGS optimization
        # model = models.CNNModel(structural_params=params, clip_loss=clip_loss, **dynamic_kwargs)
        # trainer = ProgressiveTrainer(model, max_iterations, resize_num=3)
        # ds_history = trainer.train(LBFGS_Optimizer)

        # # Example 3: CNNModel with pixel refinement using PixelRefineTrainer
        model = models.CNNModel(structural_params=params, clip_loss=clip_loss, **dynamic_kwargs)
        trainer = PixelRefineTrainer(model, max_iterations, resize_num=4, switch_threshold=200, coarse_start = False)
        ds_history = trainer.train(LBFGS_Optimizer)

        if not isinstance(ds_history, (list, np.ndarray)):
            ds_history = [ds_history]
            
        # Ensure each dataset has a step dimension for design
        for i, ds in enumerate(ds_history):
            if 'step' not in ds.design.dims:
                # Add step dimension to design if it doesn't exist
                ds = ds.expand_dims(step=[0])
                ds_history[i] = ds
            
        print(f"\nOptimization completed!")
        print(f"Number of stages: {len(ds_history)}")

        # Create and save all plots efficiently
        print("\nCreating and saving plots...")
        
        # Create loss comparison plot efficiently
        print("Creating loss comparison plot...")
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        
        # Process all datasets at once to reduce memory overhead
        for i, ds in enumerate(ds_history):
            # Rename step to iteration for consistency
            ds_renamed = ds.rename({'step': 'iteration'})
            loss_df = ds_renamed.loss.to_pandas().T
            loss_df.cummin().plot(linewidth=2, label=f"Stage {i+1}: {ds.sizes['y']}x{ds.sizes['x']}", ax=ax_loss)
            # Clear reference to reduce memory usage
            del ds_renamed, loss_df
        
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Optimization Step")
        ax_loss.set_title("Loss Comparison Across Stages")
        ax_loss.grid(True)
        ax_loss.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
        seaborn.despine()
        plt.tight_layout()
        
        # Save and display loss plot
        loss_plot_path = f'script/test_results_pytorch/optimization_comparison_loss{filename_suffix}.png'
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.show()  # Display directly without saving/loading
        plt.close(fig_loss)

        # Create final designs comparison plot efficiently
        print("Creating final designs plot...")
        fig_designs, axes = plt.subplots(1, len(ds_history), figsize=(4*len(ds_history), 6))
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        fig_designs.suptitle(f'Final Designs: {params.problem_name}', fontsize=16)

        # Get problem object for proper rendering (cache to avoid repeated lookups)
        problem = problems.PROBLEMS_BY_NAME.get(params.problem_name)
        
        # Pre-compute final designs to reduce repeated computation
        final_designs = []
        for ds in ds_history:
            final_design = ds.design.isel(step=ds.loss.argmin())
            final_designs.append(final_design)
        
        for i, (ax, final_design) in enumerate(zip(axes, final_designs)):
            if problem:
                # Use optimized numpy array version for better performance
                try:
                    design_array = pipeline_utils.image_from_design_array(final_design, problem)
                    # Invert the image for proper visualization (material = white, void = black)
                    ax.imshow(1.0 - design_array, cmap='gray')
                except:
                    # Fallback to original PIL version if needed
                    image = pipeline_utils.image_from_design(final_design, problem)
                    # Invert the image for proper visualization
                    ax.imshow(1.0 - np.array(image), cmap='gray')
            else:
                # Direct plotting for fallback with inversion
                ax.imshow(1.0 - final_design.values, cmap='gray')
            
            ax.set_title(f'Stage {i+1}: {ds_history[i].sizes["y"]}x{ds_history[i].sizes["x"]}')
            ax.axis('off')

        plt.tight_layout()
        
        # Save designs plot
        designs_plot_path = f'script/test_results_pytorch/final_designs{filename_suffix}.png'
        plt.savefig(designs_plot_path, dpi=150, bbox_inches='tight')
        plt.show()  # Display directly without saving/loading
        plt.close(fig_designs)
        
        print("All plots saved and displayed successfully!")
        print(f"Results saved to script/test_results_pytorch/")
        print(f"Files saved with suffix: {filename_suffix}")

    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())