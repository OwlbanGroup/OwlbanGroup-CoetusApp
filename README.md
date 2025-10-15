# OwlbanGroup-CoetusApp: E2E NVIDIA Blackwell AI System

This project implements an end-to-end AI system optimized for NVIDIA Blackwell GPUs, featuring image classification using pre-trained models, data preprocessing, and a FastAPI-based API for inference.

## Features

- **GPU Acceleration**: Leverages NVIDIA Blackwell GPUs for high-performance AI computations.
- **Image Classification**: Uses ResNet50 pre-trained model for ImageNet classification.
- **API Interface**: FastAPI-based REST API for easy integration.
- **Containerized Deployment**: Docker support for GPU-enabled containers.
- **Modular Design**: Separated utilities for data handling, model management, and inference.

## Prerequisites

- NVIDIA Blackwell GPU with CUDA 12.4+ support
- Docker (for containerized deployment)
- Python 3.8+ (for local development)

## Installation

### Option 1: Docker Deployment (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t blackwell-ai-system .
   ```

2. Run the container with GPU support:
   ```bash
   docker run --gpus all -p 8000:8000 blackwell-ai-system
   ```

### Option 2: Local Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure CUDA is available:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### API Usage

Start the API server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`.

#### Classify an Image

Upload an image file to classify:

```bash
curl -X POST "http://localhost:8000/classify" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```

Response:
```json
{
  "predicted_class": "golden retriever"
}
```

### Command-Line Usage

Classify an image directly:

```bash
python main.py path/to/image.jpg
```

## Project Structure

- `main.py`: Command-line interface for image classification
- `app.py`: FastAPI application for API-based inference
- `model.py`: Model loading and prediction utilities
- `data_utils.py`: Data preprocessing and utility functions
- `requirements.txt`: Python dependencies
- `Dockerfile`: Containerization configuration

## GPU Verification

To verify GPU usage:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
```

## Contributing

Contributions are welcome! Please ensure all changes are tested with GPU acceleration.

## License

This project is licensed under the MIT License.
