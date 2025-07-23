#!/usr/bin/env python3
"""
Example usage of the refactored neural structural optimization code.
This script demonstrates how to run a simple topology optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_structural_optimization import models, problems, train

def run_simple_optimization():
    """Run a simple topology optimization example."""
    print("Running simple topology optimization example...")
    
    # Create a cantilever beam problem
    problem = problems.cantilever_beam_full(
        width=40, height=20, density=0.4, force_position=0.5
    )
    print(f"Problem: {problem.name} ({problem.width}x{problem.height})")
    
    # Create a pixel model
    model = models.PixelModel(args=problem)
    print(f"Model created with {len(model.trainable_variables)} trainable variables")
    
    # Run optimization with Adam
    print("Starting optimization with Adam...")
    result = train.train_adam(model, max_iterations=50)
    
    print(f"Optimization completed!")
    print(f"Final loss: {result.loss.values[-1]:.6f}")
    print(f"Best loss: {result.loss.values.min():.6f}")
    
    return model, result

def visualize_results(model, result):
    """Visualize the optimization results."""
    print("\nVisualizing results...")
    
    # Get the best design
    best_design = result.design.values
    
    # Create a simple visualization
    plt.figure(figsize=(12, 4))
    
    # Plot loss history
    plt.subplot(1, 3, 1)
    plt.plot(result.loss.values)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot initial design
    plt.subplot(1, 3, 2)
    plt.imshow(result.design.values[0], cmap='gray', vmin=0, vmax=1)
    plt.title('Initial Design')
    plt.axis('off')
    
    # Plot final design
    plt.subplot(1, 3, 3)
    plt.imshow(best_design, cmap='gray', vmin=0, vmax=1)
    plt.title('Final Design')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'optimization_results.png'")
    
    return best_design

def compare_optimizers():
    """Compare different optimization methods."""
    print("\nComparing different optimization methods...")
    
    # Create a simple problem
    problem = problems.mbb_beam(width=30, height=15, density=0.5)
    
    # Test different optimizers
    optimizers = [
        ("Adam", lambda m: train.train_adam(m, max_iterations=30)),
        ("L-BFGS", lambda m: train.train_lbfgs(m, max_iterations=30)),
        ("Optimality Criteria", lambda m: train.optimality_criteria(m, max_iterations=30)),
    ]
    
    results = {}
    for name, optimizer_func in optimizers:
        print(f"Running {name}...")
        try:
            model = models.PixelModel(args=problem)
            result = optimizer_func(model)
            final_loss = result.loss.values[-1]
            results[name] = final_loss
            print(f"  {name} final loss: {final_loss:.6f}")
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = None
    
    # Print comparison
    print("\nOptimizer Comparison:")
    print("-" * 40)
    for name, loss in results.items():
        if loss is not None:
            print(f"{name:20s}: {loss:.6f}")
        else:
            print(f"{name:20s}: Failed")

def main():
    """Run the example."""
    print("=" * 60)
    print("Neural Structural Optimization - Example Usage")
    print("=" * 60)
    
    try:
        # Run simple optimization
        model, result = run_simple_optimization()
        
        # Visualize results
        best_design = visualize_results(model, result)
        
        # Compare optimizers
        compare_optimizers()
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 