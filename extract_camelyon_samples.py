# extract_camelyon_samples.py
import h5py
import numpy as np
from PIL import Image
import os
import random

def extract_camelyon_samples(num_samples=20):
    """Extract real images from Camelyon dataset for testing"""
    print("📁 Extracting real Camelyon samples...")
    
    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    # Path to your Camelyon files
    data_dir = 'data/raw/camelyon17'
    val_x_file = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    val_y_file = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')
    
    if not os.path.exists(val_x_file) or not os.path.exists(val_y_file):
        print("❌ Camelyon validation files not found!")
        print(f"   Looking for: {val_x_file}")
        return False
    
    try:
        with h5py.File(val_x_file, 'r') as x_file, h5py.File(val_y_file, 'r') as y_file:
            # Get total number of samples
            total_samples = len(x_file['x'])
            print(f"Found {total_samples} validation samples")
            
            # Get labels to balance our selection
            labels = y_file['y'][:total_samples]
            
            # Count classes
            normal_indices = np.where(labels == 0)[0]
            cancer_indices = np.where(labels == 1)[0]
            
            print(f"Normal samples: {len(normal_indices)}")
            print(f"Cancer samples: {len(cancer_indices)}")
            
            # Select balanced samples
            num_each = num_samples // 2
            selected_normal = random.sample(list(normal_indices), min(num_each, len(normal_indices)))
            selected_cancer = random.sample(list(cancer_indices), min(num_each, len(cancer_indices)))
            
            selected_indices = selected_normal + selected_cancer
            random.shuffle(selected_indices)
            
            # Extract and save images
            saved_count = 0
            for idx in selected_indices:
                # Get image and label
                image_data = x_file['x'][idx]
                label = y_file['y'][idx][0]
                
                # Convert to PIL Image and save
                image = Image.fromarray(image_data.astype('uint8'))
                
                class_name = 'normal' if label == 0 else 'cancer'
                filename = f'test_images/{class_name}_sample_{saved_count+1}.png'
                image.save(filename)
                
                print(f"✅ Saved: {filename}")
                saved_count += 1
            
            print(f"\n🎉 Successfully extracted {saved_count} real Camelyon images!")
            return True
            
    except Exception as e:
        print(f"❌ Error extracting samples: {e}")
        return False

def create_realistic_synthetic_images():
    """Create more realistic synthetic histopathology images"""
    import numpy as np
    from PIL import Image, ImageDraw
    
    print("🎨 Creating realistic synthetic histopathology images...")
    
    os.makedirs('test_images', exist_ok=True)
    
    for i in range(10):
        # Create base image with H&E stain colors (pink and blue)
        width, height = 224, 224
        image = Image.new('RGB', (width, height), color=(240, 230, 230))  # Light pink background
        draw = ImageDraw.Draw(image)
        
        if i < 5:
            # Normal tissue - uniform patterns
            class_name = "normal"
            # Add regular cell-like patterns
            for _ in range(50):
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(2, 8)
                color = (200 + random.randint(-20, 20), 
                        180 + random.randint(-20, 20), 
                        200 + random.randint(-20, 20))  # Purple tones
                draw.ellipse([x, y, x+radius, y+radius], fill=color)
                
        else:
            # Cancer tissue - irregular patterns
            class_name = "cancer"
            # Add darker, irregular regions
            for _ in range(3):  # Fewer but larger irregular areas
                center_x = random.randint(50, width-50)
                center_y = random.randint(50, height-50)
                size = random.randint(30, 80)
                
                # Create irregular shape (polygon)
                points = []
                for _ in range(6):
                    angle = random.uniform(0, 2 * 3.14159)
                    radius = size * random.uniform(0.7, 1.3)
                    px = center_x + radius * np.cos(angle)
                    py = center_y + radius * np.sin(angle)
                    points.append((px, py))
                
                # Darker color for cancer regions
                color = (150 + random.randint(-30, 30), 
                        120 + random.randint(-30, 30), 
                        150 + random.randint(-30, 30))
                draw.polygon(points, fill=color)
            
            # Add some abnormal cells
            for _ in range(30):
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(3, 10)
                # Darker, more variable cells
                color = (100 + random.randint(-40, 40), 
                        80 + random.randint(-40, 40), 
                        120 + random.randint(-40, 40))
                draw.ellipse([x, y, x+radius, y+radius], fill=color)
        
        filename = f'test_images/{class_name}_synthetic_{i+1}.png'
        image.save(filename)
        print(f"✅ Created: {filename}")
    
    print("\n📁 Created 10 synthetic histopathology images")

def main():
    print("🩺 CANCER DIAGNOSIS TEST IMAGES")
    print("=" * 50)
    
    # First try to extract real samples from Camelyon
    print("1. Trying to extract real Camelyon samples...")
    success = extract_camelyon_samples(10)
    
    if not success:
        print("\n2. Creating realistic synthetic images instead...")
        create_realistic_synthetic_images()
    
    print("\n🎯 Now test your model with:")
    print("   python src/predict.py --image_dir test_images/")
    print("   OR")
    print("   python app.py")
    
    # Show what's in the test_images directory
    if os.path.exists('test_images'):
        images = [f for f in os.listdir('test_images') if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\n📂 Test images available: {len(images)} files")
        for img in images[:5]:  # Show first 5
            print(f"   - {img}")

if __name__ == "__main__":
    main()