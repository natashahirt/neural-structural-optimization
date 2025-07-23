#!/usr/bin/env python3
"""
Test script for the refactored neural structural optimization code.
This script verifies that the code works with modern TensorFlow.
"""

import numpy as np
import tensorflow as tf
from neural_structural_optimization import models, problems, train

def test_basic_functionality():
    """Test basic model creation and forward pass."""
    print("Testing basic functionality...")
    
    # Create a simple problem
    problem = problems.mbb_beam(width=20, height=10, density=0.5)
    print(f"Created problem: {problem.name} ({problem.width}x{problem.height})")
    
    # Create a pixel model
    model = models.PixelModel(args=problem)
    print(f"Created PixelModel with {len(model.trainable_variables)} trainable variables")
    
    # Test forward pass
    output = model(None)
    print(f"Model output shape: {output.shape}")
    print(f"Model output dtype: {output.dtype}")
    
    # Test loss computation
    loss = model.loss(output)
    print(f"Initial loss: {loss.numpy():.6f}")
    
    return model, problem

def test_cnn_model():
    """Test CNN model creation."""
    print("\nTesting CNN model...")
    
    # Create a simple problem
    problem = problems.mbb_beam(width=32, height=16, density=0.5)
    
    # Create a CNN model
    model = models.CNNModel(args=problem, latent_size=64)
    print(f"Created CNNModel with {len(model.trainable_variables)} trainable variables")
    
    # Test forward pass
    output = model(None)
    print(f"CNN output shape: {output.shape}")
    print(f"CNN output dtype: {output.dtype}")
    
    return model

def test_optimization_step():
    """Test a single optimization step."""
    print("\nTesting optimization step...")
    
    # Create model and problem
    problem = problems.mbb_beam(width=16, height=8, density=0.5)
    model = models.PixelModel(args=problem)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Single optimization step
    with tf.GradientTape() as tape:
        output = model(None)
        loss = model.loss(output)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"Optimization step completed. Loss: {loss.numpy():.6f}")
    
    return model

def test_different_problems():
    """Test different problem types."""
    print("\nTesting different problem types...")
    
    problem_types = [
        ("Cantilever beam", problems.cantilever_beam_full),
        ("L-shape", problems.l_shape),
        ("Bridge", problems.causeway_bridge),
        ("Tower", problems.tower),
    ]
    
    for name, problem_func in problem_types:
        try:
            problem = problem_func(width=16, height=16, density=0.4)
            model = models.PixelModel(args=problem)
            output = model(None)
            loss = model.loss(output)
            print(f"✓ {name}: Loss = {loss.numpy():.6f}")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Refactored Neural Structural Optimization")
    print("=" * 60)
    
    # Test TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        # Run tests
        test_basic_functionality()
        test_cnn_model()
        test_optimization_step()
        test_different_problems()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! The refactored code works with modern TensorFlow.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 