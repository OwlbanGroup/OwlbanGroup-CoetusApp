from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from data_utils import load_image, preprocess_image, get_image_classes
from model import load_model
import io

app = FastAPI(title="NVIDIA Blackwell AI Classifier", description="End-to-end AI system using NVIDIA Blackwell GPUs")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and classes
model = load_model(device)
class_names = get_image_classes()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image using the AI model.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        image = load_image(io.BytesIO(contents))

        # Preprocess the image
        input_batch = preprocess_image(image, device)

        # Perform prediction
        from model import predict
        predicted_class = predict(model, input_batch, class_names)

        return JSONResponse(content={"predicted_class": predicted_class}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/")
async def root():
    return {"message": "NVIDIA Blackwell AI Classifier API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
