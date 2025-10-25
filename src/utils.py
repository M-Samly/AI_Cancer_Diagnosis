# src/utils.py
import os
import yaml
import torch
import random
import numpy as np
from PIL import Image

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_sample_dataset(data_dir="data/sample"):
    """Create a small sample dataset for testing"""
    import os
    from PIL import Image
    import numpy as np
    
    # Create directories
    os.makedirs(os.path.join(data_dir, "train/cancer"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "train/normal"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val/cancer"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val/normal"), exist_ok=True)
    
    print(f"Creating sample dataset in {data_dir}...")
    
    # Create more realistic sample images
    def create_medical_like_image(size, is_cancer):
        """Create synthetic medical images that look somewhat realistic"""
        # Base image - slightly textured background
        img_array = np.random.normal(128, 20, (size, size, 3)).astype(np.uint8)
        
        if is_cancer:
            # Add some "abnormal" patterns - darker regions with different textures
            for _ in range(np.random.randint(2, 5)):
                center_x = np.random.randint(0, size)
                center_y = np.random.randint(0, size)
                radius = np.random.randint(10, 30)
                
                # Create circular "abnormality"
                y, x = np.ogrid[-center_y:size-center_y, -center_x:size-center_x]
                mask = x*x + y*y <= radius*radius
                
                # Darker, more textured region for cancer
                img_array[mask] = np.clip(img_array[mask] - np.random.randint(30, 80), 0, 255)
                
                # Add some noise/texture
                texture = np.random.normal(0, 15, img_array[mask].shape).astype(np.uint8)
                img_array[mask] = np.clip(img_array[mask] + texture, 0, 255)
        else:
            # Normal tissue - more uniform with slight variations
            img_array += np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    # Create sample images for each split and class
    splits = ['train', 'val']
    classes = ['cancer', 'normal']
    
    for split in splits:
        for class_name in classes:
            is_cancer = (class_name == 'cancer')
            num_images = 20 if split == 'train' else 10
            
            for i in range(num_images):
                # Create medical-like image
                img = create_medical_like_image(128, is_cancer)
                
                filename = f"{class_name}_{i:03d}.png"
                filepath = os.path.join(data_dir, split, class_name, filename)
                img.save(filepath)
    
    print(f"Sample dataset created in {data_dir}")
    print("Directory structure:")
    print(f"  {data_dir}/train/cancer/ - 20 images")
    print(f"  {data_dir}/train/normal/ - 20 images") 
    print(f"  {data_dir}/val/cancer/ - 10 images")
    print(f"  {data_dir}/val/normal/ - 10 images")
    return True

def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path="configs/config.yaml"):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("No GPU available, using CPU")
        return False

def cleanup_sample_data(data_dir="data/sample"):
    """Clean up sample data directory"""
    import shutil
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Cleaned up {data_dir}")
    return True

if __name__ == "__main__":
    set_seed(42)
    create_sample_dataset()
    check_gpu()