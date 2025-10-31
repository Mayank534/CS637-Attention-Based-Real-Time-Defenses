"""
Minimal BiSeNet loader for testing attention defense.
This is a simplified version that creates a small CNN to simulate the BiSeNet architecture.
"""
import cv2
import torch
import torch.nn as nn

class MinimalBiSeNet(nn.Module):
    def __init__(self, num_classes=19):  # 19 classes like Cityscapes
        super().__init__()
        # Simplified backbone with early features for attention
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Rest of the network (simplified)
        self.features = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
    def forward(self, x):
        x = self.layer1(x)  # Early features for attention
        x = self.features(x)
        return x

def load_bisenet_and_preprocess():
    """Returns (model, preprocess_fn) for testing."""
    model = MinimalBiSeNet()
    
    def preprocess_fn(img_bgr):
        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        # Convert to float and normalize to [0,1]
        img_t = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
        # Add batch dimension
        img_t = img_t.unsqueeze(0)
        return img_t, {"input_hw": (H, W)}
    
    return model, preprocess_fn

def load_model():
    """Returns a minimal BiSeNet-like model for testing."""
    model = MinimalBiSeNet()
    return model

def get_shallow_module(model):
    """Returns the early layer to hook for attention."""
    return model.layer1