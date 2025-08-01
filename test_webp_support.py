#!/usr/bin/env python3
"""
Test script to verify WebP support in loop_cut_video.py
"""

from PIL import Image
import numpy as np
import os

def create_test_webp():
    """Create a simple animated WebP for testing."""
    frames = []
    # Create 10 frames with a simple animation
    for i in range(10):
        # Create a simple gradient that shifts
        img_array = np.zeros((64, 64, 3), dtype=np.uint8)
        # Moving red square
        x = (i * 5) % 64
        img_array[20:40, x:min(x+20, 64)] = [255, 0, 0]
        frames.append(Image.fromarray(img_array))
    
    # Save as animated WebP
    output_path = "test_animation.webp"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    print(f"Created test WebP: {output_path}")
    return output_path

def test_webp_processing():
    """Test the WebP processing functionality."""
    from scripts.loop_cut_video import cut_loop
    
    # Create test WebP
    test_file = create_test_webp()
    
    try:
        # Test processing
        result = cut_loop(
            input_path=test_file,
            out="processed_loop.webp",
            threshold=10  # Be more lenient for test
        )
        
        if result:
            print("WebP processing successful!")
            print(f"Results: {result}")
        else:
            print("WebP processing failed!")
            
    except Exception as e:
        print(f"Error during WebP processing: {e}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists("processed_loop.webp"):
        print("Output WebP created successfully!")
        # Keep the output for manual inspection
    
if __name__ == "__main__":
    test_webp_processing()
