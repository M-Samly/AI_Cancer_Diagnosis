# quick_test.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import CancerDiagnosisPredictor

def quick_test():
    """Quick test of the prediction system"""
    print("🚀 QUICK TEST - CANCER DIAGNOSIS AI")
    print("=" * 50)
    
    # Initialize predictor
    predictor = CancerDiagnosisPredictor()
    
    # Check test images directory
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"❌ Test directory '{test_dir}' not found.")
        print("Please run: python extract_camelyon_samples.py")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No images found in '{test_dir}'")
        return
    
    print(f"🔍 Found {len(image_files)} test images")
    print("Testing first 3 images...\n")
    
    # Test first 3 images
    for i, image_file in enumerate(image_files[:3]):
        image_path = os.path.join(test_dir, image_file)
        print(f"🧪 Testing {i+1}/3: {image_file}")
        
        result = predictor.predict_single_image(image_path)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            status = "🟡 REJECTED" if result['rejected'] else "🟢 ACCEPTED"
            prediction = result['predicted_class']
            confidence = result['confidence']
            
            print(f"   {status} - {prediction} (Confidence: {confidence:.3f})")
        
        print()

if __name__ == "__main__":
    quick_test()