#!/usr/bin/env python3
"""Test CLIP installation and functionality."""

def test_clip_installation():
    """Test if CLIP is properly installed."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        import clip
        print("✓ CLIP imported successfully")
        
        # Test loading a model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✓ CLIP model loaded on {device}")
        
        # Test tokenization
        text = clip.tokenize(["a test prompt"]).to(device)
        print("✓ Text tokenization works")
        
        # Test image preprocessing
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_input = preprocess(dummy_image).unsqueeze(0).to(device)
        print("✓ Image preprocessing works")
        
        # Test encoding
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text)
        print("✓ Feature encoding works")
        
        print("\n🎉 CLIP is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTo install CLIP, run:")
        print("pip install torch torchvision")
        print("pip install ftfy regex")
        print("pip install git+https://github.com/openai/CLIP.git")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_clip_installation() 