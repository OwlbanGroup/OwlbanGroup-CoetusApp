import sys
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from data_utils import load_image, preprocess_image, get_image_classes
from model import load_model, predict

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Load model and classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = load_model(device)
        class_names = get_image_classes()

        # Load and preprocess image
        image = load_image(image_path)
        input_batch = preprocess_image(image, device)

        # Perform prediction
        predicted_class = predict(model, input_batch, class_names)

        print(f"Predicted class: {predicted_class}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
