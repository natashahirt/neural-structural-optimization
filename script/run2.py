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

from PIL import Image
import seaborn
import matplotlib.pyplot as plt
import xarray
import pandas as pd
import numpy as np

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train
from neural_structural_optimization import pipeline_utils

def train_all(problem, max_iterations, cnn_kwargs=None):
    """Train all optimization methods on a given problem."""
    print(f"Starting optimization for {problem.name} ({problem.width}x{problem.height})")
    
    args = topo_api.specified_task(problem)

    if cnn_kwargs is None:
        cnn_kwargs = {}

    print("Running dynamic CNN Model with L-BFGS...")
    model = models.CNNModelDynamic(args=args)
    ds_cnn = train.train_lbfgs(model, max_iterations)

    # resolutions = [
    #     (20, 40),   # coarse
    #     (40, 80),   # mid
    #     (60, 120),  # fine
    # ]

    # ds_cnn = train.adaptive_train_lbfgs(
    #     args,
    #     resolutions=resolutions,
    #     max_iter_per_stage=200,
    # )

    dims = pd.Index(['cnn-lbfgs'], name='model')
    result = xarray.concat([ds_cnn], dim=dims)
    return result

def main():
    """Main function with error handling and progress reporting."""
    print("=" * 60)
    print("Neural Structural Optimization - Multi-Method Comparison")
    print("=" * 60)
    
    try:
        # Create problem
        problem = problems.multistory_building(width=5, height=15, interval=3)
        max_iterations = 200
        
        print(f"Problem: {problem.name}")
        print(f"Dimensions: {problem.width}x{problem.height}")
        print(f"Max iterations: {max_iterations}")

        # Run all optimization methods
        print("\nStarting optimization...")

        # args = topo_api.specified_task(problem)
        # model = models.CNNModelDynamic(args=args)
        # ds = train.train_lbfgs(model, max_iterations)

        ds_history = train.adaptive_train_lbfgs(problem, [(5,15),(10,20),(20,60)], max_iterations)
        if not isinstance(ds_history, list):
            ds_history = [ds_history]
            
        print(f"\nOptimization completed!")
        print(f"Number of stages: {len(ds_history)}")

        # Create loss comparison plot
        print("\nCreating loss comparison plot...")
        plt.figure(figsize=(10, 6))
        for i, ds in enumerate(ds_history):
            ds = ds.rename({'step': 'iteration'})
            loss_df = ds.loss.to_pandas().T
            loss_df.cummin().plot(linewidth=2, label=f"Stage {i+1}: {ds.dims['y']}x{ds.dims['x']}")
        plt.ylabel("Loss")
        plt.xlabel("Optimization Step")
        plt.title("Loss Comparison Across Stages")
        plt.grid(True)
        plt.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
        seaborn.despine()
        plt.tight_layout()
        plt.savefig('optimization_comparison_loss.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Create final designs comparison plot
        print("\nCreating final designs plot...")
        fig, axes = plt.subplots(1, len(ds_history), figsize=(4*len(ds_history), 6))
        fig.suptitle(f'Final Designs: {problem.name}', fontsize=16)

        for i, (ax, ds) in enumerate(zip(axes, ds_history)):
            ds = ds.rename({'step': 'iteration'})
            final_design = ds.design.isel(iteration=ds.loss.argmin())
            im = ax.imshow(final_design, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Stage {i+1}: {ds.dims["y"]}x{ds.dims["x"]}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'results_{problem.name}.png', dpi=150, bbox_inches='tight')
        print(f"Final designs plot saved to 'results_{problem.name}.png'")
        plt.show()

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