# src/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_coverage_curve(metrics):
    """
    Plot accuracy-coverage curve
    
    Args:
        metrics: Metrics dictionary from UncertaintyEstimator
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(metrics['coverage_levels'], metrics['accuracies'], 
             marker='o', linewidth=2, markersize=8, label='Selective Accuracy')
    
    # Add horizontal line for full accuracy
    plt.axhline(y=metrics['full_accuracy'], color='r', linestyle='--', 
                label=f'Full Accuracy ({metrics["full_accuracy"]:.3f})')
    
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Coverage Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/accuracy_coverage_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_distribution(uncertainties, labels, accepted_mask):
    """
    Plot uncertainty distribution for accepted vs rejected samples
    
    Args:
        uncertainties: Uncertainty scores
        labels: True labels
        accepted_mask: Boolean mask of accepted predictions
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Overall distribution
    plt.subplot(1, 2, 1)
    plt.hist(uncertainties, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution by acceptance
    plt.subplot(1, 2, 2)
    accepted_uncertainties = uncertainties[accepted_mask]
    rejected_uncertainties = uncertainties[~accepted_mask]
    
    plt.hist(accepted_uncertainties, bins=30, alpha=0.7, label='Accepted', edgecolor='black')
    plt.hist(rejected_uncertainties, bins=30, alpha=0.7, label='Rejected', edgecolor='black')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty: Accepted vs Rejected')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/uncertainty_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(metrics, uncertainties, labels, accepted_mask, save_path='results/report.txt'):
    """
    Generate comprehensive evaluation report
    
    Args:
        metrics: Calculated metrics
        uncertainties: Uncertainty scores
        labels: True labels
        accepted_mask: Boolean mask of accepted predictions
        save_path: Path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("UNCERTAINTY-AWARE CANCER DIAGNOSIS EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Full Dataset Accuracy: {metrics['full_accuracy']:.4f}\n")
        f.write(f"Selective Accuracy: {metrics['selective_accuracy']:.4f}\n")
        f.write(f"Coverage Rate: {metrics['coverage_rate']:.4f}\n")
        f.write(f"Rejection Rate: {metrics['rejection_rate']:.4f}\n\n")
        
        f.write("UNCERTAINTY STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Uncertainty: {np.mean(uncertainties):.4f}\n")
        f.write(f"Std Uncertainty: {np.std(uncertainties):.4f}\n")
        f.write(f"Min Uncertainty: {np.min(uncertainties):.4f}\n")
        f.write(f"Max Uncertainty: {np.max(uncertainties):.4f}\n\n")
        
        f.write("REJECTION ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        accepted_count = np.sum(accepted_mask)
        rejected_count = np.sum(~accepted_mask)
        total_count = len(accepted_mask)
        
        f.write(f"Total Samples: {total_count}\n")
        f.write(f"Accepted Samples: {accepted_count} ({100.*accepted_count/total_count:.2f}%)\n")
        f.write(f"Rejected Samples: {rejected_count} ({100.*rejected_count/total_count:.2f}%)\n")
        
        # Accuracy by class for accepted samples
        if accepted_count > 0:
            accepted_labels = labels[accepted_mask]
            unique_classes = np.unique(labels)
            f.write("\nACCURACY BY CLASS (Accepted Samples):\n")
            for class_id in unique_classes:
                class_mask = accepted_labels == class_id
                if np.sum(class_mask) > 0:
                    class_correct = np.sum(accepted_labels[class_mask] == class_id)
                    class_acc = class_correct / np.sum(class_mask)
                    class_name = "Cancer" if class_id == 1 else "Normal"
                    f.write(f"  {class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)\n")

def test_evaluation():
    """Test evaluation functions with sample data"""
    print("Testing evaluation functions...")
    
    # Create sample data
    sample_metrics = {
        'full_accuracy': 0.85,
        'selective_accuracy': 0.92,
        'coverage_rate': 0.7,
        'rejection_rate': 0.3,
        'coverage_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'accuracies': [0.95, 0.94, 0.93, 0.93, 0.92, 0.91, 0.90, 0.88, 0.86, 0.85]
    }
    
    sample_uncertainties = np.random.beta(2, 5, 1000)
    sample_labels = np.random.randint(0, 2, 1000)
    sample_accepted = sample_uncertainties < 0.5
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Test plotting functions
    plot_accuracy_coverage_curve(sample_metrics)
    plot_uncertainty_distribution(sample_uncertainties, sample_labels, sample_accepted)
    generate_report(sample_metrics, sample_uncertainties, sample_labels, sample_accepted)
    
    print("Evaluation tests completed. Check 'results' directory for output files.")
    return True

if __name__ == "__main__":
    test_evaluation()