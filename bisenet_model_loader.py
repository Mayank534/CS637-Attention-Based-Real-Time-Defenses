# """
# Minimal BiSeNet loader for testing attention defense.
# This is a simplified version that creates a small CNN to simulate the BiSeNet architecture.
# """
# import cv2
# import torch
# import torch.nn as nn

# class MinimalBiSeNet(nn.Module):
#     def __init__(self, num_classes=19):  # 19 classes like Cityscapes
#         super().__init__()
#         # Simplified backbone with early features for attention
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, 7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         # Rest of the network (simplified)
#         self.features = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, num_classes, 1)
#         )
        
#     def forward(self, x):
#         x = self.layer1(x)  # Early features for attention
#         x = self.features(x)
#         return x

# def load_bisenet_and_preprocess():
#     """Returns (model, preprocess_fn) for testing."""
#     model = MinimalBiSeNet()
    
#     def preprocess_fn(img_bgr):
#         # Convert BGR to RGB and normalize
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         H, W = img_rgb.shape[:2]
#         # Convert to float and normalize to [0,1]
#         img_t = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
#         # Add batch dimension
#         img_t = img_t.unsqueeze(0)
#         return img_t, {"input_hw": (H, W)}
    
#     return model, preprocess_fn

# def load_model():
#     """Returns a minimal BiSeNet-like model for testing."""
#     model = MinimalBiSeNet()
#     return model

# def get_shallow_module(model):
#     """Returns the early layer to hook for attention."""
#     return model.layer1


"""
BiSeNet-like minimal backbone with explicit shallow/tail split
so we can (optionally) avoid a full second forward by resuming
from the shallow feature after masking (single-pass friendly).

This is a compact stand-in for the real BiSeNet. Swap it with
your actual model later keeping the same interface:
 - forward(x) -> logits
 - forward_from_shallow(feat) -> logits
 - get_shallow_module(model) -> nn.Module to hook
 - load_bisenet_and_preprocess() -> (model, preprocess_fn)
"""

import cv2
import torch
import torch.nn as nn

class MinimalBiSeNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        # Shallow (hook here)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # A small "context" block (play the role of deeper stages)
        self.context = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Head
        self.head = nn.Conv2d(128, num_classes, 1, bias=True)

    def forward(self, x):
        # full forward
        shallow = self.layer1(x)           # (B,64,H/2,W/2)
        deep = self.context(shallow)       # (B,128,·,·)
        out = self.head(deep)              # (B,classes,·,·)
        return out

    def forward_from_shallow(self, shallow_feat: torch.Tensor):
        # resume from shallow features after in-place masking
        deep = self.context(shallow_feat)
        out = self.head(deep)
        return out


# def load_bisenet_and_preprocess():
#     """
#     Returns (model, preprocess_fn)
#     preprocess_fn(img_bgr) -> (tensor[1,3,H,W], {"input_hw": (H,W)})
#     """
#     model = MinimalBiSeNet()

#     # Simple normalize to [0,1] (swap with dataset stats when you use real weights)
#     def preprocess_fn(img_bgr):
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         H, W = img_rgb.shape[:2]
#         img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
#         return img_t.unsqueeze(0), {"input_hw": (H, W)}

#     return model, preprocess_fn
import torchvision
def load_bisenet_and_preprocess():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
    def preprocess_fn(img_bgr):
        import torchvision.transforms as T
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        t = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        return t(img_rgb).unsqueeze(0), {"input_hw": (H,W)}
    return model, preprocess_fn


def load_model():
    return MinimalBiSeNet()
def get_shallow_module(model):
    """
    Return the shallow feature module to hook for attention.
    Supports:
      • custom MinimalBiSeNet (has .layer1)
      • torchvision segmentation models (DeepLabV3 / FCN) (have .backbone)
    """
    # Case 1: Your own dummy BiSeNet
    if hasattr(model, "layer1"):
        return model.layer1
    # Case 2: Torchvision segmentation model
    elif hasattr(model, "backbone"):
        # Most torchvision segmentation models (DeepLabV3, FCN)
        # use a ResNet backbone accessible here
        backbone = model.backbone
        # ResNet layers: conv1, bn1, relu, maxpool, layer1, layer2, ...
        if "layer1" in backbone:
            return backbone["layer1"]
        elif hasattr(backbone, "layer1"):
            return backbone.layer1
        else:
            raise AttributeError("Cannot find 'layer1' inside DeepLabV3 backbone.")
    else:
        raise AttributeError(f"Cannot find shallow layer in model of type {type(model)}")
