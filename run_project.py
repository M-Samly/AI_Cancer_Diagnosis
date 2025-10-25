# run_project.py
import os
import sys
import subprocess
import threading
import time
import webbrowser
from src.utils import set_seed

def check_trained_model():
    """Check if model is trained, if not train it"""
    model_path = 'saved_models/best_model.pth'
    
    if not os.path.exists(model_path):
        print("🤖 No trained model found. Starting training...")
        print("This may take a while. Please wait...")
        
        # Train the model
        try:
            result = subprocess.run([
                sys.executable, 'main.py', '--mode', 'train',
                '--dataset_type', 'camelyon',
                '--data_dir', 'data/raw/camelyon17',
                '--epochs', '5',  # Quick training for demo
                '--batch_size', '16',
                '--max_samples', '2000'  # Use smaller subset for quick training
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Model training completed!")
                return True
            else:
                print(f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    else:
        print("✅ Trained model found!")
        return True

def start_web_server():
    """Start the Flask web server"""
    print("🚀 Starting web server...")
    os.system('python app.py')

def open_browser():
    """Open web browser after a delay"""
    time.sleep(3)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

def main():
    set_seed(42)
    
    print("🎯 CANCER DIAGNOSIS AI PROJECT")
    print("=" * 50)
    
    # Check and train model if needed
    if not check_trained_model():
        print("❌ Cannot start project without trained model.")
        return
    
    print("\n📊 Project Options:")
    print("1. 🕸️  Web Interface (Recommended)")
    print("2. 💻 Command Line Interface")
    print("3. 🔧 Development Mode")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == "1":
        # Web interface
        print("\nStarting web interface...")
        print("The browser will open automatically.")
        print("If not, manually go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server.")
        
        # Start browser in separate thread
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start web server (this will block)
        start_web_server()
        
    elif choice == "2":
        # Command line interface
        print("\nStarting command line interface...")
        print("Available commands:")
        print("  python src/predict.py --image_path path/to/image.png")
        print("  python src/predict.py --image_dir test_images/")
        print("  python main.py --mode evaluate --dataset_type camelyon")
        
        # Run a sample prediction if test images exist
        test_dir = "test_images"
        if os.path.exists(test_dir) and os.listdir(test_dir):
            print(f"\n🔍 Found images in {test_dir}. Running sample prediction...")
            os.system(f'python src/predict.py --image_dir {test_dir} --num_samples 5')
        else:
            print(f"\n📁 Please add some test images to '{test_dir}' directory.")
            print("Then run: python src/predict.py --image_dir test_images/")
            
    elif choice == "3":
        # Development mode
        print("\n🔧 Development Mode - Available Scripts:")
        print("  Training: python main.py --mode train --dataset_type camelyon")
        print("  Evaluation: python main.py --mode evaluate --dataset_type camelyon")
        print("  Testing: python main.py --mode test_camelyon")
        print("  Web Interface: python app.py")
        print("  Prediction: python src/predict.py --image_path your_image.png")
        
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main()