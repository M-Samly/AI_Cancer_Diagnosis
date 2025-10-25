# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not available. Using custom CNN architecture.")
    TORCHVISION_AVAILABLE = False

class BasicCNN(nn.Module):
    """
    Custom CNN architecture as fallback when torchvision is not available
    """
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(BasicCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class UncertaintyAwareModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5, method='mc_dropout', use_pretrained=True):
        """
        Uncertainty-aware model for cancer diagnosis
        
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for uncertainty estimation
            method (str): Uncertainty method ('mc_dropout' or 'ensemble')
            use_pretrained (bool): Whether to use pre-trained weights (if available)
        """
        super(UncertaintyAwareModel, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.method = method
        
        # Use pre-trained ResNet as backbone if available, otherwise use custom CNN
        if TORCHVISION_AVAILABLE and use_pretrained:
            print("Using pre-trained ResNet18 backbone")
            self.backbone = models.resnet18(pretrained=True)
            
            # Replace final layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original fc layer
            
            # Add dropout and new classification head
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(in_features, num_classes)
            
            self.feature_size = in_features
            
        else:
            print("Using custom CNN backbone (torchvision not available or pretrained disabled)")
            self.backbone = BasicCNN(num_classes, dropout_rate)
            self.feature_size = 512  # For the custom CNN
            
    def forward(self, x, num_samples=1):
        """
        Forward pass with optional multiple samples for uncertainty
        
        Args:
            x: Input tensor
            num_samples: Number of stochastic forward passes
        
        Returns:
            outputs: Model predictions
            uncertainties: Uncertainty estimates
        """
        
        if self.method == 'mc_dropout' and num_samples > 1:
            return self.mc_dropout_forward(x, num_samples)
        else:
            # Standard forward pass
            if TORCHVISION_AVAILABLE:
                features = self.backbone(x)
                features = self.dropout(features)
                output = self.classifier(features)
            else:
                output = self.backbone(x)
            return output, None
    
    def mc_dropout_forward(self, x, num_samples=10):
        """
        Monte Carlo Dropout forward pass
        
        Args:
            x: Input tensor
            num_samples: Number of stochastic forward passes
        
        Returns:
            mean_output: Mean prediction across samples
            uncertainty: Predictive uncertainty
        """
        # Enable dropout during inference
        self.train()
        
        outputs = []
        for _ in range(num_samples):
            if TORCHVISION_AVAILABLE:
                features = self.backbone(x)
                features = self.dropout(features)
                output = self.classifier(features)
            else:
                output = self.backbone(x)
            outputs.append(F.softmax(output, dim=1))
        
        # Stack and compute statistics
        outputs = torch.stack(outputs)  # [num_samples, batch_size, num_classes]
        mean_output = torch.mean(outputs, dim=0)
        
        # Calculate uncertainty as entropy
        uncertainty = -torch.sum(mean_output * torch.log(mean_output + 1e-10), dim=1)
        
        return mean_output, uncertainty
    
    def get_uncertainty(self, x, num_samples=10):
        """
        Convenience method to get only uncertainty
        """
        _, uncertainty = self.mc_dropout_forward(x, num_samples)
        return uncertainty

class DeepEnsemble:
    def __init__(self, num_models=5, num_classes=2, use_pretrained=True):
        """
        Deep Ensemble for uncertainty estimation
        
        Args:
            num_models (int): Number of models in ensemble
            num_classes (int): Number of output classes
            use_pretrained (bool): Whether to use pre-trained weights
        """
        self.num_models = num_models
        self.num_classes = num_classes
        self.models = []
        self.use_pretrained = use_pretrained
        
    def create_models(self):
        """Create ensemble of models"""
        for i in range(self.num_models):
            model = UncertaintyAwareModel(
                num_classes=self.num_classes,
                dropout_rate=0.3,
                method='standard',
                use_pretrained=self.use_pretrained
            )
            self.models.append(model)
    
    def predict(self, x, num_samples=1):
        """
        Get predictions from ensemble
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples per model
        
        Returns:
            mean_prediction: Mean prediction across ensemble
            uncertainty: Ensemble disagreement
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if num_samples > 1:
                    output, _ = model.mc_dropout_forward(x, num_samples)
                else:
                    output, _ = model(x)
                prob = F.softmax(output, dim=1)
                predictions.append(prob)
        
        predictions = torch.stack(predictions)  # [num_models, batch_size, num_classes]
        mean_prediction = torch.mean(predictions, dim=0)
        
        # Uncertainty as variance across models
        uncertainty = torch.var(predictions, dim=0).mean(dim=1)
        
        return mean_prediction, uncertainty

def test_models():
    """Test the model architectures"""
    print("Testing model architectures...")
    
    # Test with small input
    batch_size = 2
    img_size = 128
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    try:
        # Test UncertaintyAwareModel
        print("Testing UncertaintyAwareModel...")
        model = UncertaintyAwareModel(num_classes=2, dropout_rate=0.5)
        
        # Standard forward
        output, _ = model(x)
        print(f"Output shape: {output.shape}")
        
        # MC Dropout forward
        mean_output, uncertainty = model.mc_dropout_forward(x, num_samples=5)
        print(f"MC Dropout output shape: {mean_output.shape}")
        print(f"Uncertainty shape: {uncertainty.shape}")
        
        # Test DeepEnsemble
        print("\nTesting DeepEnsemble...")
        ensemble = DeepEnsemble(num_models=3, num_classes=2)
        ensemble.create_models()
        
        mean_pred, ensemble_uncertainty = ensemble.predict(x)
        print(f"Ensemble prediction shape: {mean_pred.shape}")
        print(f"Ensemble uncertainty shape: {ensemble_uncertainty.shape}")
        
        print("✅ All model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_models()