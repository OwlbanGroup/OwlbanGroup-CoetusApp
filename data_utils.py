"""Image loading and preprocessing utilities for the classifier."""
import os
from PIL import Image
from torchvision import transforms


def load_image(image_path):
    """
    Load an image and return it as a PIL Image.

    Accepts either a filesystem path (str/pathlib) or a file-like object
    (e.g. io.BytesIO from an uploaded file). A FileNotFoundError is raised
    for a missing path; file-like objects are opened directly.
    """
    if isinstance(image_path, (str, bytes, os.PathLike)):
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


_IMAGENET_CLASSES_PATH = os.path.join(os.path.dirname(__file__),
                                      "imagenet_classes.txt")


def get_image_classes():
    """
    Return 1000 ImageNet class labels aligned with the model's softmax output.

    Reads ``imagenet_classes.txt`` (the standard PyTorch Hub label list)
    shipped next to this module. Falls back to ``class_0..class_999``
    placeholders when the file is missing, so prediction always resolves.
    """
    try:
        with open(_IMAGENET_CLASSES_PATH, encoding="utf-8") as fh:
            labels = [line.strip() for line in fh if line.strip()]
        if len(labels) == 1000:
            return labels
    except OSError:
        pass
    return [f"class_{i}" for i in range(1000)]
