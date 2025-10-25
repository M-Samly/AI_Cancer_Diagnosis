# src/predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys

# Fix import path - add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our models
from src.models import UncertaintyAwareModel

class CancerDiagnosisPredictor:
    def __init__(self, model_path='saved_models/best_model.pth', uncertainty_threshold=0.4):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path (str): Path to trained model weights
            uncertainty_threshold (float): Threshold for rejection
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = UncertaintyAwareModel(
            num_classes=2,
            dropout_rate=0.5,
            method='mc_dropout'
        )
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️  No trained model found at {model_path}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations - using 96x96 for Camelyon dataset
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Camelyon images are 96x96
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Normal', 'Cancer']
    
    def predict_single_image(self, image_path, num_samples=10):
        """
        Predict a single image with uncertainty estimation
        
        Args:
            image_path (str): Path to image file
            num_samples (int): Number of MC Dropout samples
        
        Returns:
            dict: Prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Transform image to 96x96 (Camelyon size)
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Get prediction with uncertainty
            with torch.no_grad():
                mean_prediction, uncertainty = self.model.mc_dropout_forward(
                    input_tensor, num_samples=num_samples
                )
            
            # Convert to probabilities
            probabilities = F.softmax(mean_prediction[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            
            # Decision with rejection
            uncertainty_value = uncertainty[0].item()
            confidence_value = confidence.item()
            predicted_class = predicted_class.item()
            
            if uncertainty_value > self.uncertainty_threshold:
                decision = "REJECT - Refer to Human Expert"
                final_class = "Uncertain"
            else:
                decision = "ACCEPT - Model Prediction"
                final_class = self.class_names[predicted_class]
            
            results = {
                'image_path': image_path,
                'original_size': original_size,
                'predicted_class': self.class_names[predicted_class],
                'final_decision': final_class,
                'decision_reason': decision,
                'confidence': confidence_value,
                'uncertainty': uncertainty_value,
                'normal_probability': probabilities[0].item(),
                'cancer_probability': probabilities[1].item(),
                'rejected': uncertainty_value > self.uncertainty_threshold
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f"Error processing image: {str(e)}",
                'image_path': image_path
            }
    
    def predict_batch(self, image_dir, num_samples=10):
        """
        Predict all images in a directory
        
        Args:
            image_dir (str): Directory containing images
            num_samples (int): Number of MC Dropout samples
        
        Returns:
            list: List of prediction results
        """
        results = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        
        if not os.path.exists(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            return results
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            print(f"❌ No valid images found in {image_dir}")
            return results
        
        print(f"🔍 Found {len(image_files)} images to process...")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            
            result = self.predict_single_image(image_path, num_samples)
            results.append(result)
        
        return results

def print_prediction_result(result):
    """Print formatted prediction results"""
    print("\n" + "="*60)
    print("🎯 CANCER DIAGNOSIS PREDICTION RESULTS")
    print("="*60)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📁 Image: {os.path.basename(result['image_path'])}")
    print(f"📐 Original Size: {result['original_size']}")
    print(f"🎯 Predicted Class: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.4f}")
    print(f"❓ Uncertainty: {result['uncertainty']:.4f}")
    print(f"⚖️  Decision: {result['decision_reason']}")
    print(f"✅ Final Decision: {result['final_decision']}")
    print(f"📈 Probabilities:")
    print(f"   - Normal: {result['normal_probability']:.4f}")
    print(f"   - Cancer: {result['cancer_probability']:.4f}")
    
    if result['rejected']:
        print("🚨 HIGH UNCERTAINTY - Referring to human expert")
    else:
        if result['predicted_class'] == 'Cancer':
            print("⚠️  FOLLOW-UP RECOMMENDED - Further examination needed")
        else:
            print("✅ LOW RISK - No immediate action required")

def save_predictions_to_file(results, output_file='predictions_report.txt'):
    """Save prediction results to a text file"""
    with open(output_file, 'w') as f:
        f.write("CANCER DIAGNOSIS PREDICTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results):
            if 'error' in result:
                f.write(f"Image {i+1}: {result['image_path']}\n")
                f.write(f"Error: {result['error']}\n\n")
                continue
            
            f.write(f"Image {i+1}: {os.path.basename(result['image_path'])}\n")
            f.write(f"Predicted Class: {result['predicted_class']}\n")
            f.write(f"Confidence: {result['confidence']:.4f}\n")
            f.write(f"Uncertainty: {result['uncertainty']:.4f}\n")
            f.write(f"Decision: {result['decision_reason']}\n")
            f.write(f"Normal Probability: {result['normal_probability']:.4f}\n")
            f.write(f"Cancer Probability: {result['cancer_probability']:.4f}\n")
            
            if result['rejected']:
                f.write("STATUS: REJECTED - High uncertainty, human review needed\n")
            else:
                f.write("STATUS: ACCEPTED - Model prediction\n")
            
            f.write("-" * 40 + "\n\n")
    
    print(f"📄 Prediction report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Cancer Diagnosis Prediction')
    parser.add_argument('--image_path', type=str, help='Path to single image file')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--model_path', type=str, default='saved_models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.4,
                       help='Uncertainty threshold for rejection')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of MC Dropout samples')
    parser.add_argument('--output_file', type=str, default='predictions_report.txt',
                       help='Output file for batch predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CancerDiagnosisPredictor(
        model_path=args.model_path,
        uncertainty_threshold=args.uncertainty_threshold
    )
    
    if args.image_path:
        # Single image prediction
        if os.path.exists(args.image_path):
            result = predictor.predict_single_image(args.image_path, args.num_samples)
            print_prediction_result(result)
        else:
            print(f"❌ Image not found: {args.image_path}")
    
    elif args.image_dir:
        # Batch prediction
        results = predictor.predict_batch(args.image_dir, args.num_samples)
        
        # Print summary
        total = len(results)
        accepted = sum(1 for r in results if not r.get('rejected', True))
        rejected = total - accepted
        
        print(f"\n📊 BATCH PREDICTION SUMMARY:")
        print(f"   Total images: {total}")
        print(f"   Accepted predictions: {accepted}")
        print(f"   Rejected (high uncertainty): {rejected}")
        if total > 0:
            print(f"   Rejection rate: {rejected/total*100:.1f}%")
        
        # Save to file
        save_predictions_to_file(results, args.output_file)
        
        # Show detailed results for first few images
        print(f"\n📋 DETAILED RESULTS (first 5 images):")
        for i, result in enumerate(results[:5]):
            if 'error' not in result:
                status = "REJECTED" if result['rejected'] else result['predicted_class']
                print(f"   {i+1}. {os.path.basename(result['image_path'])}: {status}")
    
    else:
        print("❌ Please specify either --image_path or --image_dir")
        print("\nUsage examples:")
        print("  Single image: python src/predict.py --image_path path/to/image.png")
        print("  Batch images: python src/predict.py --image_dir path/to/images/")
        print("  Custom model: python src/predict.py --image_path image.png --model_path saved_models/retrained_model.pth")
        print("  Adjust uncertainty: python src/predict.py --image_path image.png --uncertainty_threshold 0.3")

if __name__ == "__main__":
    main()