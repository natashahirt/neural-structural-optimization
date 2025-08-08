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

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train
from neural_structural_optimization import pipeline_utils
from neural_structural_optimization.problems_utils import ProblemParams
from neural_structural_optimization.clip_loss import CLIPLoss

def create_filename_suffix(suffix_str):
    if not suffix_str:
        return ""

    suffix = suffix_str.replace(' ', '_').replace('=', '_').replace(',', '_')
    suffix = suffix.replace('"', '').replace("'", '')
    suffix = suffix.replace('/', '_').replace('\\', '_')
    suffix = suffix.replace(':', '_').replace(';', '_')

    suffix = re.sub(r'[^a-zA-Z0-9_]', '', suffix)

    print(f"Using filename suffix: {suffix}")
    
    return f"_{suffix}"

def main():
    """Main function with error handling and progress reporting."""
    print("=" * 60)
    print("Neural Structural Optimization - Multi-Method Comparison")
    print("=" * 60)

    suffix_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    filename_suffix = create_filename_suffix(suffix_str)        
    
    try:
        # Create problem
        max_iterations = 200

        # Run all optimization methods
        print("\nStarting optimization...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_loss = CLIPLoss(
            device=device,
            model_name="ViT-B/32",
            target_text_prompt="skeleton",
            weight=1.0,                     # start small; tune upward
        )

        # note that width and height are targets and not absolute
        params = ProblemParams(
            problem_name = "multistory_building",
            width=30, # 50
            height=100, # 40
            density=0.3,
            num_stories=8
        )

        params, dynamic_kwargs = pipeline_utils.dynamic_depth_kwargs(params)      

        print("Dynamic kwargs:")
        for key, value in dynamic_kwargs.items():
            print(f"  {key}: {value}")

        print(f"Problem: {params.problem_name}")
        print(f"Dimensions: {params.width}x{params.height}")
        print(f"Max iterations: {max_iterations}")

        model = models.PixelModel(problem_params=params, clip_loss=clip_loss)
        ds_history = train.train_lbfgs(model, max_iterations)

        # model = models.CNNModel(problem_params=params, **dynamic_kwargs)
        # ds_history = train.train_lbfgs(model, max_iterations)

        # model = models.PixelModel(problem_params=params, resize_num=2, clip_config=clip_config)
        # ds_history = train.train_progressive(model, max_iterations, alg=train.train_lbfgs)

        # model = models.CNNModel(problem_params=params, clip_config=clip_config, resize_num=3, activation=tf.nn.relu, **dynamic_kwargs)
        # ds_history = train.train_progressive(model, max_iterations, alg=train.train_lbfgs)

        if not isinstance(ds_history, (list, np.ndarray)):
            ds_history = [ds_history]
            
        print(f"\nOptimization completed!")
        print(f"Number of stages: {len(ds_history)}")

        # Create loss comparison plot
        print("\nCreating loss comparison plot...")
        plt.figure(figsize=(10, 6))
        for i, ds in enumerate(ds_history):
            ds = ds.rename({'step': 'iteration'})
            loss_df = ds.loss.to_pandas().T
            loss_df.cummin().plot(linewidth=2, label=f"Stage {i+1}: {ds.sizes['y']}x{ds.sizes['x']}")
        plt.ylabel("Loss")
        plt.xlabel("Optimization Step")
        plt.title("Loss Comparison Across Stages")
        plt.grid(True)
        plt.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
        seaborn.despine()
        plt.tight_layout()
        plt.savefig(f'script/test_results_pytorch/optimization_comparison_loss{filename_suffix}.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Create final designs comparison plot
        print("\nCreating final designs plot...")
        fig, axes = plt.subplots(1, len(ds_history), figsize=(4*len(ds_history), 6))
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        fig.suptitle(f'Final Designs: {params.problem_name}', fontsize=16)

        for i, (ax, ds) in enumerate(zip(axes, ds_history)):
            final_design = ds.design.isel(step=ds.loss.argmin())
            im = ax.imshow(1 - final_design, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Stage {i+1}: {ds.sizes["y"]}x{ds.sizes["x"]}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'script/test_results_pytorch/results_{params.problem_name}{filename_suffix}.png', dpi=150, bbox_inches='tight')
        print(f"Final designs plot saved to 'results_{params.problem_name}{filename_suffix}.png'")
        plt.show()

        # Save final high-resolution design separately
        final_ds = ds_history[-1]
        final_design = final_ds.design.isel(step=final_ds.loss.argmin())
        plt.figure(figsize=(8, 8))
        plt.imshow(1 - final_design, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'Final Design ({final_ds.sizes["y"]}x{final_ds.sizes["x"]})', pad=20)
        plt.savefig(f'script/test_results_pytorch/final_design_{params.problem_name}{filename_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"High-resolution final design saved to 'final_design_{params.problem_name}{filename_suffix}.png'")

        print("\n" + "=" * 60)
        print("All operations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nScript failed. Check the error message above.")

if __name__ == "__main__":
    main()