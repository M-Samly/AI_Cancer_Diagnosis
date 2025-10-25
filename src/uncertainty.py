# src/uncertainty.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

class UncertaintyEstimator:
    def __init__(self, method='mc_dropout', threshold=0.3):
        """
        Uncertainty estimation and rejection mechanism
        
        Args:
            method (str): Uncertainty estimation method
            threshold (float): Uncertainty threshold for rejection
        """
        self.method = method
        self.threshold = threshold
        
    def calculate_uncertainty(self, model, data_loader, num_samples=10):
        """
        Calculate uncertainty for a dataset
        
        Args:
            model: Trained model
            data_loader: DataLoader for the dataset
            num_samples: Number of MC samples
        
        Returns:
            predictions: Model predictions
            uncertainties: Uncertainty scores
            labels: True labels
        """
        model.eval()
        
        all_predictions = []
        all_uncertainties = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(data_loader):
                if self.method == 'mc_dropout':
                    outputs, uncertainties = model(images, num_samples=num_samples)
                else:
                    outputs, uncertainties = model(images)
                
                # Convert to probabilities
                probs = F.softmax(outputs, dim=1)
                
                # If no uncertainty from model, calculate entropy
                if uncertainties is None:
                    uncertainties = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                
                all_predictions.extend(probs.cpu().numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_paths.extend(paths)
                
                if batch_idx % 10 == 0:
                    print(f'Processed batch {batch_idx}/{len(data_loader)}')
        
        return (np.array(all_predictions), 
                np.array(all_uncertainties), 
                np.array(all_labels),
                all_paths)
    
    def apply_rejection(self, predictions, uncertainties, labels):
        """
        Apply rejection based on uncertainty threshold
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty scores
            labels: True labels
        
        Returns:
            accepted_indices: Indices of accepted predictions
            rejected_indices: Indices of rejected predictions
        """
        accepted_mask = uncertainties < self.threshold
        accepted_indices = np.where(accepted_mask)[0]
        rejected_indices = np.where(~accepted_mask)[0]
        
        return accepted_indices, rejected_indices
    
    def calculate_metrics(self, predictions, uncertainties, labels):
        """
        Calculate accuracy-coverage metrics
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty scores
            labels: True labels
        
        Returns:
            metrics_dict: Dictionary of calculated metrics
        """
        pred_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics at different coverage levels
        coverage_levels = np.linspace(0.1, 1.0, 10)
        accuracies = []
        coverages = []
        
        for coverage in coverage_levels:
            # Find uncertainty threshold for this coverage
            threshold = np.percentile(uncertainties, coverage * 100)
            accepted_mask = uncertainties <= threshold
            
            if np.sum(accepted_mask) > 0:
                coverage_acc = np.mean(pred_classes[accepted_mask] == labels[accepted_mask])
            else:
                coverage_acc = 0.0
                
            accuracies.append(coverage_acc)
            coverages.append(coverage)
        
        # Full dataset accuracy (no rejection)
        full_accuracy = np.mean(pred_classes == labels)
        
        # With current threshold
        accepted_mask = uncertainties < self.threshold
        if np.sum(accepted_mask) > 0:
            selective_accuracy = np.mean(pred_classes[accepted_mask] == labels[accepted_mask])
            coverage_rate = np.mean(accepted_mask)
        else:
            selective_accuracy = 0.0
            coverage_rate = 0.0
        
        metrics = {
            'full_accuracy': full_accuracy,
            'selective_accuracy': selective_accuracy,
            'coverage_rate': coverage_rate,
            'rejection_rate': 1 - coverage_rate,
            'coverage_levels': coverages,
            'accuracies': accuracies
        }
        
        return metrics

def test_uncertainty():
    """Test uncertainty estimation"""
    print("Testing uncertainty estimation...")
    
    # Create sample data
    sample_predictions = np.random.rand(100, 2)
    sample_predictions = sample_predictions / np.sum(sample_predictions, axis=1, keepdims=True)
    sample_uncertainties = -np.sum(sample_predictions * np.log(sample_predictions + 1e-10), axis=1)
    sample_labels = np.random.randint(0, 2, 100)
    
    estimator = UncertaintyEstimator(threshold=0.5)
    metrics = estimator.calculate_metrics(sample_predictions, sample_uncertainties, sample_labels)
    
    print("Uncertainty metrics test completed")
    print(f"Full accuracy: {metrics['full_accuracy']:.3f}")
    print(f"Selective accuracy: {metrics['selective_accuracy']:.3f}")
    print(f"Coverage rate: {metrics['coverage_rate']:.3f}")
    
    return True

if __name__ == "__main__":
    test_uncertainty()