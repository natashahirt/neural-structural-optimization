#!/usr/bin/env python3
"""
Example usage of the refactored neural structural optimization code.
This script demonstrates how to run a simple topology optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_structural_optimization import models, problems, train, topo_api
from neural_structural_optimization import clip_config

def run_pixel_optimization(problem=problems.cantilever_beam_full(width=40,height=20,density=.4,force_position=0.5)):
    """Run a simple topology optimization example."""
    print("Running simple topology optimization example...")
    
    # Create a cantilever beam problem
    print(f"Problem: {problem.name} ({problem.width}x{problem.height})")

    args = topo_api.specified_task(problem)
    
    # Create a pixel model
    model = models.PixelModel(args=args)
    print(f"Model created with {len(model.trainable_variables)} trainable variables")
    
    # Run optimization with Adam - much more iterations for better resolution
    print("Starting optimization with Adam...")
    result = train.train_adam(model, max_iterations=1000)  # Increased from 200 to 500
    
    print(f"Optimization completed!")
    print(f"Final loss: {result.loss.values[-1]:.6f}")
    print(f"Best loss: {result.loss.values.min():.6f}")
    print(f"Total iterations: {len(result.loss.values)}")
    
    return model, result
    
def visualize_results(model, result, filename=""):
    """Visualize the optimization results."""
    print("\nVisualizing results...")
    
    # Get the best design - handle both 2D and 3D arrays
    best_design = result.design.values
    if len(best_design.shape) == 3:
        # Take final design if we have intermediate steps
        best_design = best_design[-1]
    
    # Create a simple visualization
    plt.figure(figsize=(12, 4))
    
    # Plot loss history
    plt.subplot(1, 3, 1)
    plt.plot(result.loss.values)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot initial design - handle both 2D and 3D arrays
    plt.subplot(1, 3, 2)
    initial_design = result.design.values[0] if len(result.design.values.shape) == 3 else result.design.values
    plt.imshow(initial_design, cmap='gray', vmin=0, vmax=1)
    plt.title('Initial Design')
    plt.axis('off')
    
    # Plot final design
    plt.subplot(1, 3, 3)
    plt.imshow(best_design, cmap='gray', vmin=0, vmax=1)
    plt.title('Final Design')
    plt.axis('off')
    plt.tight_layout()
    if filename != "":
        plt.savefig(f'optimization_results_{filename}.png', dpi=150, bbox_inches='tight')
        print(f"Results saved to 'optimization_results_{filename}.png'")
    else:
        plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
        print("Results saved to 'optimization_results.png'")
    
    return best_design

def run_cnn_optimization():
    """Run optimization using the CNN model."""
    print("Running CNN-based optimization...")
    
    # Create a cantilever beam problem
    problem = problems.cantilever_beam_full(
        width=80, height=40, density=0.4, force_position=0.5
    )
    print(f"Problem: {problem.name} ({problem.width}x{problem.height})")

    args = topo_api.specified_task(problem)
    
    # Create a CNN model instead of PixelModel
    model = models.CNNModel(
        args=args,
        latent_size=128,  # Size of latent space
    )
    print(f"CNN Model created with {len(model.trainable_variables)} trainable variables")
    
    # Run optimization with Adam
    print("Starting optimization with Adam...")
    result = train.train_adam(model, max_iterations=500)
    
    print(f"Optimization completed!")
    print(f"Final loss: {result.loss.values[-1]:.6f}")
    print(f"Best loss: {result.loss.values.min():.6f}")
    print(f"Total iterations: {len(result.loss.values)}")
    
    return model, result

def run_clip_cnn_optimization(prompt="structure"):
    """Run CNN optimization with CLIP guidance."""
    print("Running CNN optimization with CLIP guidance...")
    
    # Create a cantilever beam problem
    problem = problems.multistory_building(
        width=50, height=100
    )
    print(f"Problem: {problem.name} ({problem.width}x{problem.height})")
    
    # Create CLIP configuration
    clip_cfg = clip_config.create_clip_config(
        target_text_prompt=prompt,
        weight=1.0 # Balance between physics and aesthetics
    )
    
    # Create CNN model with CLIP
    model = models.CNNModel(
        args=problem,
        clip_config=clip_cfg,  # Pass CLIP config
    )
    print(f"CNN Model with CLIP created with {len(model.trainable_variables)} trainable variables")
    
    # Run optimization with Adam
    print("Starting optimization with Adam...")
    result = train.train_adam(model, max_iterations=500)
    
    print(f"Optimization completed!")
    print(f"Final loss: {result.loss.values[-1]:.6f}")
    print(f"Best loss: {result.loss.values.min():.6f}")
    print(f"Total iterations: {len(result.loss.values)}")
    
    return model, result

def main():
    """Run the example."""
    print("=" * 60)
    print("Neural Structural Optimization - Example Usage")
    print("=" * 60)
    
    problem = problems.multistory_building(width=50,height=100)
    
    try:
        # Run simple optimization with more iterations
        model, result = run_pixel_optimization(problem=problem)
        best_design = visualize_results(model, result, filename=f"pixelmodel_{problem.name}")

        # Run CNN optimization
        # cnn_model, cnn_result = run_clip_cnn_optimization(prompt="human skeleton")
        # cnn_design = visualize_results(cnn_model, cnn_result, filename="clip")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 