#!/usr/bin/env python3
"""
Ultra-fast setup script for Neural Structural Optimization.

This script bypasses pip entirely and just sets up the Python path
so you can import the package immediately.

Usage:
    python setup.py
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Add current directory to Python path."""
    current_dir = Path.cwd().absolute()
    print(f"Adding {current_dir} to Python path...")
    
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print("[SUCCESS] Directory added to Python path")
    else:
        print("[INFO] Directory already in Python path")
    
    return True

def verify_installation():
    """Verify that the package can be imported."""
    print("Verifying installation...")
    
    try:
        import neural_structural_optimization
        print("[SUCCESS] Package import successful!")
        
        # Test importing key modules
        from neural_structural_optimization import models, problems, train
        print("[SUCCESS] Key modules imported successfully!")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import verification failed: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Try installing dependencies first:")
        print("  python -m pip install -r requirements.txt")
        return False

def create_activation_script():
    """Create a script to activate the environment."""
    script_content = '''@echo off
REM Activation script for Neural Structural Optimization
echo Setting up Python path for neural_structural_optimization...

set PYTHONPATH=%CD%;%PYTHONPATH%

echo Environment activated!
echo You can now run: python script/run.py
echo.
'''
    
    with open('activate_env.bat', 'w') as f:
        f.write(script_content)
    
    print("[SUCCESS] Created activation script: activate_env.bat")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Neural Structural Optimization - Ultra-Fast Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("neural_structural_optimization").exists():
        print("[ERROR] Please run this script from the project root directory")
        print("   (where the 'neural_structural_optimization' folder is located)")
        sys.exit(1)
    
    # Setup Python path
    if not setup_python_path():
        sys.exit(1)
    
    # Create activation script
    create_activation_script()
    
    # Verify installation
    if not verify_installation():
        print("\n[WARNING] Package import failed, but setup completed.")
        print("You may need to install dependencies:")
        print("  python -m pip install -r requirements.txt")
        print("\nOr use the activation script:")
        print("  activate_env.bat")
        sys.exit(1)
    
    print("\n[SUCCESS] Setup completed successfully!")
    print("\nYou can now run:")
    print("  python script/run.py")
    print("\nOr use the activation script:")
    print("  activate_env.bat")

if __name__ == "__main__":
    main()

# Minimal setuptools configuration for compatibility
import setuptools

setuptools.setup(
    name='neural-structural-optimization',
    version='0.1.0',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    description='Neural reparameterization for structural optimization',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
