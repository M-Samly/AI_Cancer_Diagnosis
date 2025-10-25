# app.py
from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from src.models import UncertaintyAwareModel
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UncertaintyAwareModel(num_classes=2, dropout_rate=0.5, method='mc_dropout')
model_path = 'saved_models/best_model.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully")
else:
    print("⚠️  Using untrained model")

model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def predict_image(image, uncertainty_threshold=0.4, num_samples=10):
    """Predict a single image"""
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction with uncertainty
    with torch.no_grad():
        mean_prediction, uncertainty = model.mc_dropout_forward(
            input_tensor, num_samples=num_samples
        )
    
    # Convert to probabilities
    probabilities = F.softmax(mean_prediction[0], dim=0)
    confidence, predicted_class = torch.max(probabilities, 0)
    
    uncertainty_value = uncertainty[0].item()
    confidence_value = confidence.item()
    predicted_class = predicted_class.item()
    
    # Decision
    if uncertainty_value > uncertainty_threshold:
        decision = "REJECT - High Uncertainty"
        final_class = "Uncertain"
        color = "warning"
    else:
        decision = "ACCEPT - Model Prediction"
        final_class = "Cancer" if predicted_class == 1 else "Normal"
        color = "danger" if predicted_class == 1 else "success"
    
    return {
        'predicted_class': final_class,
        'confidence': round(confidence_value, 4),
        'uncertainty': round(uncertainty_value, 4),
        'normal_probability': round(probabilities[0].item(), 4),
        'cancer_probability': round(probabilities[1].item(), 4),
        'decision': decision,
        'color': color,
        'rejected': uncertainty_value > uncertainty_threshold
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        try:
            # Read image
            image = Image.open(file.stream).convert('RGB')
            
            # Save original image for display
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            img_data = base64.b64encode(img_io.getvalue()).decode()
            
            # Get prediction
            result = predict_image(image)
            result['image_data'] = f"data:image/png;base64,{img_data}"
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)