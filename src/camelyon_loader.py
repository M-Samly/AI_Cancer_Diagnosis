# src/camelyon_loader.py
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CamelyonDataset(Dataset):
    def __init__(self, x_file, y_file, transform=None, max_samples=None):
        """
        Camelyon17 dataset loader for HDF5 files
        
        Args:
            x_file (str): Path to HDF5 file containing images
            y_file (str): Path to HDF5 file containing labels
            transform: Image transformations
            max_samples (int): Maximum number of samples to load (for testing)
        """
        self.x_file = x_file
        self.y_file = y_file
        self.transform = transform
        
        # Open HDF5 files to get dataset info
        with h5py.File(self.x_file, 'r') as f:
            self.num_samples = len(f['x'])
            if max_samples is not None:
                self.num_samples = min(self.num_samples, max_samples)
        
        print(f"Loaded Camelyon dataset with {self.num_samples} samples from {x_file}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load image from HDF5 file
        with h5py.File(self.x_file, 'r') as f:
            image_data = f['x'][idx]  # Shape: (96, 96, 3) for Camelyon
            
        # Load label from HDF5 file
        with h5py.File(self.y_file, 'r') as f:
            label = f['y'][idx]  # Shape: (1,)
            label = int(label[0])  # Convert to integer
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_data.astype('uint8'))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, f"sample_{idx}"

def get_camelyon_data_loaders(data_dir, batch_size=32, img_size=96, max_samples=None):
    """
    Create data loaders for Camelyon17 dataset
    
    Args:
        data_dir (str): Path to directory containing HDF5 files
        batch_size (int): Batch size
        img_size (int): Image size for resizing (Camelyon patches are 96x96)
        max_samples (int): Maximum samples per split (for testing)
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    
    # File paths
    train_x = os.path.join(data_dir, 'camelyonpatch_level_2_split_train_x.h5')
    train_y = os.path.join(data_dir, 'camelyonpatch_level_2_split_train_y.h5')
    valid_x = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    valid_y = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')
    test_x = os.path.join(data_dir, 'camelyonpatch_level_2_split_test_x.h5')
    test_y = os.path.join(data_dir, 'camelyonpatch_level_2_split_test_y.h5')
    
    # Verify files exist
    for file_path in [train_x, train_y, valid_x, valid_y, test_x, test_y]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Camelyon data file not found: {file_path}")
    
    # Data transforms - note: Camelyon images are 96x96
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
    train_dataset = CamelyonDataset(
        x_file=train_x,
        y_file=train_y,
        transform=train_transform,
        max_samples=max_samples
    )
    
    val_dataset = CamelyonDataset(
        x_file=valid_x,
        y_file=valid_y,
        transform=val_transform,
        max_samples=max_samples
    )
    
    test_dataset = CamelyonDataset(
        x_file=test_x,
        y_file=test_y,
        transform=val_transform,
        max_samples=max_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Print dataset info
    print(f"Camelyon dataset loaded:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Check class distribution
    with h5py.File(train_y, 'r') as f:
        train_labels = f['y'][:len(train_dataset)]
        print(f"  Class distribution - Normal: {np.sum(train_labels == 0)}, Tumor: {np.sum(train_labels == 1)}")
    
    return train_loader, val_loader, test_loader

def test_camelyon_loader():
    """Test the Camelyon data loader"""
    print("Testing Camelyon data loader...")
    
    data_dir = "data/raw/camelyon17"
    
    try:
        # Load a small subset for testing
        train_loader, val_loader, test_loader = get_camelyon_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            img_size=96,
            max_samples=100  # Only load 100 samples per split for testing
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Check one batch
        for images, labels, paths in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
            
        return True
        
    except Exception as e:
        print(f"Error in Camelyon data loader: {e}")
        return False

if __name__ == "__main__":
    test_camelyon_loader()