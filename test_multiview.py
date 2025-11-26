"""
Quick test script to verify multi-view 3D reconstruction works.
This creates a simple test to ensure the code runs without errors.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PIL import Image
import numpy as np

def create_test_images(num_images=3):
    """Create simple test images with gradients."""
    images = []
    for i in range(num_images):
        # Create a gradient image with slight variations
        img_array = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Create different patterns for each image
        for y in range(400):
            for x in range(400):
                # Gradient based on position and image index
                r = int((x / 400) * 255)
                g = int((y / 400) * 255)
                b = int(((x + y) / 800) * 255 + i * 30) % 255
                img_array[y, x] = [r, g, b]
        
        images.append(Image.fromarray(img_array))
    
    return images

def test_multiview():
    """Test the multi-view reconstruction."""
    print("Creating test images...")
    images = create_test_images(3)
    
    print("Testing multi-view reconstruction...")
    from multiview_to_mesh import multiview_to_mesh
    
    output_path = "generated_meshes/test_multiview.obj"
    os.makedirs("generated_meshes", exist_ok=True)
    
    try:
        result = multiview_to_mesh(images, output_path, poisson_depth=8)
        print(f"\n✓ Test passed! Mesh saved to: {result}")
        return True
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multiview()
    sys.exit(0 if success else 1)
