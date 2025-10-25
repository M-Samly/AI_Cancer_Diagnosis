# diagnose_model.py
import torch
import numpy as np
from src.models import UncertaintyAwareModel
from src.camelyon_loader import get_camelyon_data_loaders
import os

def test_model_on_validation_set():
    """Test model on actual validation data to see real performance"""
    print("🧪 Testing Model on Validation Data...")
    
    # Load model
    model = UncertaintyAwareModel(num_classes=2, dropout_rate=0.5)
    model_path = 'saved_models/best_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Get validation data
    _, val_loader, _ = get_camelyon_data_loaders(
        data_dir='data/raw/camelyon17',
        batch_size=32,
        img_size=96,
        max_samples=1000
    )
    
    # Test on validation set
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = 100 * correct / total
    print(f"📊 Validation Accuracy: {accuracy:.2f}%")
    
    # Detailed analysis
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    cancer_predictions = np.sum(predictions == 1)
    cancer_actual = np.sum(labels == 1)
    
    print(f"🎯 Predictions: {cancer_predictions}/{total} cancer ({cancer_predictions/total*100:.1f}%)")
    print(f"📈 Actual: {cancer_actual}/{total} cancer ({cancer_actual/total*100:.1f}%)")
    
    bias = cancer_predictions / total
    if bias > 0.8:
        print("🚨 CRITICAL: Model is severely biased towards cancer predictions!")
        return False
    elif accuracy < 60:
        print("⚠️  WARNING: Model accuracy is very low!")
        return False
    else:
        print("✅ Model performance seems reasonable")
        return True

def check_model_outputs():
    """Check what the model is actually outputting"""
    print("\n🔍 Analyzing Model Outputs...")
    
    model = UncertaintyAwareModel(num_classes=2, dropout_rate=0.5)
    model_path = 'saved_models/best_model.pth'
    
    if not os.path.exists(model_path):
        return
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Test with random input
    test_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output, _ = model(test_input)
        probabilities = torch.softmax(output, dim=1)
    
    print(f"📊 Raw output: {output[0].numpy()}")
    print(f"📈 Probabilities: Normal={probabilities[0][0].item():.4f}, Cancer={probabilities[0][1].item():.4f}")
    
    # Check if outputs are extreme
    if probabilities[0][1] > 0.9:
        print("🚨 Model is extremely confident in cancer class!")

def analyze_test_images():
    """Analyze predictions on test images with details"""
    print("\n🖼️  Analyzing Test Images...")
    
    from src.predict import CancerDiagnosisPredictor
    
    predictor = CancerDiagnosisPredictor()
    
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print("❌ Test images directory not found")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} test images")
    
    cancer_count = 0
    normal_count = 0
    rejected_count = 0
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        result = predictor.predict_single_image(image_path, num_samples=5)
        
        if 'error' in result:
            print(f"❌ {image_file}: Error - {result['error']}")
            continue
        
        prediction = result['predicted_class']
        confidence = result['confidence']
        uncertainty = result['uncertainty']
        
        if result['rejected']:
            status = "REJECTED"
            rejected_count += 1
        elif prediction == 'Cancer':
            status = "CANCER"
            cancer_count += 1
        else:
            status = "NORMAL"
            normal_count += 1
        
        print(f"📁 {image_file}: {status} (Conf: {confidence:.3f}, Uncert: {uncertainty:.3f})")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Normal: {normal_count}")
    print(f"   Cancer: {cancer_count}")
    print(f"   Rejected: {rejected_count}")
    
    if cancer_count > normal_count * 3:  # If more than 75% are cancer
        print("🚨 BIAS DETECTED: Model strongly favors cancer predictions!")

if __name__ == "__main__":
    print("🩺 COMPREHENSIVE MODEL DIAGNOSIS")
    print("=" * 60)
    
    # Test on validation data
    validation_ok = test_model_on_validation_set()
    
    # Check model outputs
    check_model_outputs()
    
    # Analyze test images
    analyze_test_images()
    
    if not validation_ok:
        print("\n💡 SOLUTION: The model needs to be retrained with better parameters.")
        print("   Run: python retrain_model.py")