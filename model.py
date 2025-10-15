import torch
from torchvision.models import resnet50

def load_model(device='cpu'):
    """
    Load a pre-trained ResNet50 model and move it to the specified device.
    """
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
