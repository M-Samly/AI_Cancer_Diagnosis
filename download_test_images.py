# download_test_images.py
import os
import urllib.request
import zipfile

def download_camelyon_samples():
    """Download sample Camelyon images for testing"""
    print("Downloading sample test images...")
    
    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    # Sample images from Camelyon dataset (these are small examples)
    sample_urls = [
        "https://github.com/BMIRDS/deepslide/raw/master/examples/positive/1.png",
        "https://github.com/BMIRDS/deepslide/raw/master/examples/positive/2.png", 
        "https://github.com/BMIRDS/deepslide/raw/master/examples/negative/1.png",
        "https://github.com/BMIRDS/deepslide/raw/master/examples/negative/2.png"
    ]
    
    for i, url in enumerate(sample_urls):
        try:
            filename = f"test_images/sample_{i+1}.png"
            urllib.request.urlretrieve(url, filename)
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
    
    print("\n📁 Sample images saved to 'test_images/' directory")
    print("These are proper histopathology images for testing.")

def create_sample_images():
    """Create synthetic histopathology-like images for testing"""
    import numpy as np
    from PIL import Image
    
    print("Creating synthetic test images...")
    
    os.makedirs('test_images', exist_ok=True)
    
    # Create synthetic images that resemble histopathology
    for i in range(6):
        # Create image with histopathology-like patterns
        img_array = np.random.normal(128, 30, (224, 224, 3)).astype(np.uint8)
        
        # Add some "tissue-like" patterns
        if i < 3:
            # Normal tissue - more uniform
            img_array += np.random.normal(0, 15, img_array.shape).astype(np.uint8)
            label = "normal"
        else:
            # Cancer-like - add darker irregular regions
            center_x, center_y = np.random.randint(50, 174, 2)
            radius = np.random.randint(20, 40)
            
            # Create circular "abnormality"
            y, x = np.ogrid[-center_y:224-center_y, -center_x:224-center_x]
            mask = x*x + y*y <= radius*radius
            
            # Darker region for cancer-like appearance
            img_array[mask] = np.clip(img_array[mask] - np.random.randint(40, 80), 0, 255)
            label = "cancer"
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        filename = f"test_images/{label}_{i+1}.png"
        img.save(filename)
        print(f"✅ Created: {filename}")
    
    print("\n📁 Synthetic test images created in 'test_images/'")

if __name__ == "__main__":
    print("🩺 CANCER DIAGNOSIS TEST IMAGES SETUP")
    print("=" * 50)
    
    try:
        download_camelyon_samples()
    except:
        print("❌ Could not download real samples. Creating synthetic ones...")
        create_sample_images()
    
    print("\n🎯 Now you can test with:")
    print("   python src/predict.py --image_dir test_images/")
    print("   OR")
    print("   python app.py  # then upload images via web interface")