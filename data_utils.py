import os
from PIL import Image
import torch
from torchvision import transforms

def load_image(image_path):
    """
    Load an image from the given path and return it as a PIL Image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    return Image.open(image_path)

def preprocess_image(image, device='cpu'):
    """
    Preprocess the image for model input.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch

def get_image_classes():
    """
    Return the list of ImageNet class names.
    """
    # Full ImageNet class names (1000 classes)
    return [
    'toilet tissue'
    ]
