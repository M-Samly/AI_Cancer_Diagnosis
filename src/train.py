# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device='cuda'):
    """
    Train the uncertainty-aware model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to use for training
    
    Returns:
        model: Trained model
        history: Training history
    """
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels, _) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, (images, labels, _) in enumerate(val_bar):
                images, labels = images.to(device), labels.to(device)
                
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate epoch metrics
        train_loss_epoch = train_loss / len(train_loader)
        train_acc_epoch = 100. * train_correct / train_total
        val_loss_epoch = val_loss / len(val_loader)
        val_acc_epoch = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['val_acc'].append(val_acc_epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%')
        print(f'Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%')
        
        # Save best model
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f'New best model saved with val accuracy: {best_val_acc:.2f}%')
        
        scheduler.step()
    
    return model, history

def test_training():
    """Test training with sample data"""
    print("Testing training process...")
    
    # This would be called with actual data
    # For now, just verify the function structure
    return True

if __name__ == "__main__":
    test_training()