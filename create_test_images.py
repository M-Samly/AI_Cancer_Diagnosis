# create_test_images.py
import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_simple_test_images():
    """Create simple test images that clearly show the difference"""
    print("Creating clear test images...")
    
    os.makedirs('test_images', exist_ok=True)
    
    # Clear examples that your model can distinguish
    for i in range(8):
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        if i < 4:
            # NORMAL - Light, uniform patterns
            class_name = "normal"
            # Light pink background (H&E stain - eosin)
            img = Image.new('RGB', (224, 224), color=(255, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Add some light purple nuclei (H&E stain - hematoxylin)
            for _ in range(100):
                x = random.randint(10, 214)
                y = random.randint(10, 214)
                radius = random.randint(3, 8)
                color = (200, 180, 220)  # Light purple
                draw.ellipse([x, y, x+radius, y+radius], fill=color)
                
        else:
            # CANCER - Dark, irregular patterns  
            class_name = "cancer"
            # Slightly darker background
            img = Image.new('RGB', (224, 224), color=(240, 220, 220))
            draw = ImageDraw.Draw(img)
            
            # Add dark, irregular clusters
            for cluster in range(5):
                center_x = random.randint(30, 194)
                center_y = random.randint(30, 194)
                cluster_size = random.randint(20, 60)
                
                # Irregular cluster of dark cells
                for _ in range(15):
                    x = center_x + random.randint(-cluster_size, cluster_size)
                    y = center_y + random.randint(-cluster_size, cluster_size)
                    radius = random.randint(4, 10)
                    color = (120, 80, 160)  # Dark purple
                    draw.ellipse([x, y, x+radius, y+radius], fill=color)
        
        filename = f'test_images/{class_name}_test_{i+1}.png'
        img.save(filename)
        print(f"✅ Created: {filename}")
    
    print("\n🎨 Created 8 clear test images (4 normal, 4 cancer)")

if __name__ == "__main__":
    create_simple_test_images()