"""Image classification model loading and inference helpers."""
import torch
from torchvision.models import resnet50


def load_model(device='cpu'):
    """
    Load a pre-trained ResNet50 model and move it to the specified device.

    Uses the modern torchvision weights API when available and falls back
    to the legacy `pretrained=True` flag on older torchvision releases.
    """
    try:
        from torchvision.models import ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except ImportError:  # torchvision < 0.13
        model = resnet50(pretrained=True)
    model.eval()
    model.to(device)
    return model


def predict(model, input_batch, class_names):
    """
    Perform inference on the input batch and return the predicted class.
    """
    with torch.no_grad():
        output = model(input_batch)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        return predicted_class
