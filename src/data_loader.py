# src/data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class CancerDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Dataset class for cancer histopathology images
        
        Args:
            data_dir (str): Path to dataset directory
            transform: Image transformations
            is_train (bool): Whether this is training data
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        
        if is_train:
            base_path = os.path.join(data_dir, 'train')
        else:
            base_path = os.path.join(data_dir, 'val')
            
        print(f"Looking for images in: {base_path}")
        
        # For cancer class (label 1)
        cancer_path = os.path.join(base_path, 'cancer')
        if os.path.exists(cancer_path):
            cancer_images = [f for f in os.listdir(cancer_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            for img_file in cancer_images:
                self.image_paths.append(os.path.join(cancer_path, img_file))
                self.labels.append(1)
            print(f"Found {len(cancer_images)} cancer images")
        
        # For normal class (label 0)
        normal_path = os.path.join(base_path, 'normal')
        if os.path.exists(normal_path):
            normal_images = [f for f in os.listdir(normal_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            for img_file in normal_images:
                self.image_paths.append(os.path.join(normal_path, img_file))
                self.labels.append(0)
            print(f"Found {len(normal_images)} normal images")
        
        print(f"Total images found: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def get_data_loaders(data_dir, batch_size=32, img_size=224, create_sample_if_empty=True):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size
        img_size (int): Image size for resizing
        create_sample_if_empty (bool): Create sample data if directory is empty
    """
    
    # Check if data directory exists and has images
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        if create_sample_if_empty:
            print(f"No data found in {data_dir}. Creating sample dataset...")
            from src.utils import create_sample_dataset
            create_sample_dataset(data_dir)
        else:
            raise ValueError(f"No data found in {data_dir}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CancerDataset(
        data_dir=data_dir, 
        transform=train_transform, 
        is_train=True
    )
    
    val_dataset = CancerDataset(
        data_dir=data_dir, 
        transform=val_transform, 
        is_train=False
    )
    
    # Check if we have enough data
    if len(train_dataset) == 0:
        raise ValueError("No training images found!")
    if len(val_dataset) == 0:
        raise ValueError("No validation images found!")
    
    # Adjust batch size if we have very few samples
    actual_batch_size = min(batch_size, len(train_dataset))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True, 
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=actual_batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader

def test_data_loader():
    """Test function to verify data loading works"""
    print("Testing data loader...")
    
    # Use sample data directory
    sample_dir = "data/sample"
    
    try:
        train_loader, val_loader = get_data_loaders(
            data_dir=sample_dir, 
            batch_size=4, 
            img_size=128
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Check one batch
        for images, labels, paths in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")
            print(f"Sample paths: {paths[:2]}")
            break
            
        return True
        
    except Exception as e:
        print(f"Error in data loader: {e}")
        return False

if __name__ == "__main__":
    test_data_loader()