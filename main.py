# main.py
import os
import torch
import argparse
import numpy as np
from src.data_loader import get_data_loaders, test_data_loader
from src.camelyon_loader import get_camelyon_data_loaders, test_camelyon_loader  # Add this import
from src.models import UncertaintyAwareModel
from src.train import train_model
from src.uncertainty import UncertaintyEstimator
from src.evaluate import plot_training_history, plot_accuracy_coverage_curve, generate_report
from src.utils import set_seed, check_gpu, create_sample_dataset, cleanup_sample_data

def main():
    parser = argparse.ArgumentParser(description='Uncertainty-Aware Cancer Diagnosis')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'train', 'evaluate', 'full', 'cleanup', 'test_camelyon'],
                       help='Run mode: test (quick test), train, evaluate, full pipeline, cleanup, or test_camelyon')
    parser.add_argument('--data_dir', type=str, default='data/sample',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, default='folder',
                       choices=['folder', 'camelyon'],
                       help='Type of dataset: folder (regular images) or camelyon (HDF5 files)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.3,
                       help='Uncertainty threshold for rejection')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for testing)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up sample data before running')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Cleanup if requested
    if args.cleanup and args.data_dir == 'data/sample':
        cleanup_sample_data(args.data_dir)
    
    if args.mode == 'test':
        print("Running in TEST mode with sample data...")
        
        # Create sample dataset if it doesn't exist
        if not os.path.exists(args.data_dir):
            create_sample_dataset(args.data_dir)
        
        # Test data loading
        print("\n1. Testing data loader...")
        success = test_data_loader()
        if not success:
            print("❌ Data loader test failed!")
            return
        
        # Test model creation
        print("\n2. Testing model creation...")
        model = UncertaintyAwareModel(num_classes=2, dropout_rate=0.5)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test uncertainty estimation
        print("\n3. Testing uncertainty estimation...")
        from src.uncertainty import test_uncertainty
        test_uncertainty()
        
        # Test evaluation
        print("\n4. Testing evaluation...")
        from src.evaluate import test_evaluation
        test_evaluation()
        
        print("\n✅ All tests completed successfully!")
        print("You can now proceed with real data by:")
        print("1. Using Camelyon dataset: --mode train --dataset_type camelyon --data_dir data/raw/camelyon17")
        print("2. Using folder dataset: --mode train --dataset_type folder --data_dir path/to/your/data")
    
    elif args.mode == 'test_camelyon':
        print("Testing Camelyon data loader...")
        success = test_camelyon_loader()
        if success:
            print("✅ Camelyon data loader test passed!")
        else:
            print("❌ Camelyon data loader test failed!")
    
    elif args.mode == 'train':
        print(f"Training model on {args.data_dir} (type: {args.dataset_type})...")
        
        # Create data loaders based on dataset type
        if args.dataset_type == 'camelyon':
            train_loader, val_loader, _ = get_camelyon_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=96,  # Camelyon patches are 96x96
                max_samples=args.max_samples
            )
        else:  # folder type
            train_loader, val_loader = get_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=224
            )
        
        # Create model
        model = UncertaintyAwareModel(
            num_classes=2,
            dropout_rate=0.5,
            method='mc_dropout'
        )
        
        # Train model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            device=device
        )
        
        # Plot training history
        plot_training_history(history)
        print("Training completed! Check 'results' directory for plots.")
    
    elif args.mode == 'evaluate':
        print("Evaluating model...")
        
        # Load trained model
        model = UncertaintyAwareModel(num_classes=2, dropout_rate=0.5)
        model_path = 'saved_models/best_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        else:
            print(f"Warning: No trained model found at {model_path}. Using untrained model.")
        
        # Create data loader based on dataset type
        if args.dataset_type == 'camelyon':
            _, val_loader, _ = get_camelyon_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=96,
                max_samples=args.max_samples
            )
        else:
            _, val_loader = get_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=224
            )
        
        # Calculate uncertainty
        estimator = UncertaintyEstimator(
            method='mc_dropout',
            threshold=args.uncertainty_threshold
        )
        
        predictions, uncertainties, labels, paths = estimator.calculate_uncertainty(
            model, val_loader, num_samples=10
        )
        
        # Apply rejection
        accepted_indices, rejected_indices = estimator.apply_rejection(
            predictions, uncertainties, labels
        )
        
        # Calculate metrics
        metrics = estimator.calculate_metrics(predictions, uncertainties, labels)
        
        # Generate plots and report
        plot_accuracy_coverage_curve(metrics)
        
        accepted_mask = np.zeros(len(uncertainties), dtype=bool)
        accepted_mask[accepted_indices] = True
        
        generate_report(metrics, uncertainties, labels, accepted_mask)
        
        print("Evaluation completed! Check 'results' directory for output files.")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Full Accuracy: {metrics['full_accuracy']:.3f}")
        print(f"Selective Accuracy: {metrics['selective_accuracy']:.3f}")
        print(f"Coverage Rate: {metrics['coverage_rate']:.3f}")
        print(f"Rejection Rate: {metrics['rejection_rate']:.3f}")
    
    elif args.mode == 'cleanup':
        cleanup_sample_data(args.data_dir)
        print("Cleanup completed!")
    
    elif args.mode == 'full':
        print("Running full pipeline...")
        # This would run the complete training and evaluation pipeline
        pass

if __name__ == "__main__":
    main()