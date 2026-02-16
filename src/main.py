from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Spoilage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and labels variable
model = None
labels = {0: "Fresh", 1: "Rotten"} # Default

def load_data_model():
    """Load the trained model and class indices."""
    global model, labels
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spoilage_model.keras')
    indices_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_indices.json')
    
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            if os.path.exists(indices_path):
                import json
                with open(indices_path, 'r') as f:
                    class_indices = json.load(f)
                    # Invert the dictionary: {0: 'fresh', 1: 'rotten'}
                    labels = {v: k.capitalize() for k, v in class_indices.items()}
                    logger.info(f"Loaded labels: {labels}")
        except Exception as e:
            logger.error(f"Failed to load model or labels: {e}")
    else:
        logger.warning(f"Model not found at {model_path}. Prediction will fail.")

@app.on_event("startup")
async def startup_event():
    load_data_model()

def preprocess_image(image_bytes):
    """Preprocess image for MobileNetV2."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image)
        # MobileNetV2 expects input in range [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, labels
    if not model:
        if os.environ.get("MOCK_MODE") == "true":
            return {"prediction": "Fresh", "confidence": 0.88, "spoiled_percentage": 0.12, "is_spoiled": False, "note": "MOCK RESPONSE"}
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    if processed_image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        predictions = model.predict(processed_image)
        score = predictions[0]
        
        # Find which index corresponds to 'Rotten'
        rotten_idx = -1
        for idx, label in labels.items():
            if label.lower() == 'rotten':
                rotten_idx = idx
                break
        
        if rotten_idx == -1:
            # Fallback if 'rotten' not in labels (e.g. only 1 class or different name)
            # Assume index 1 is rotten if 2 classes
            rotten_idx = 1 if len(score) > 1 else 0
            
        spoiled_percentage = float(score[rotten_idx])
        
        # Determine overall label based on argmax
        predicted_class_idx = np.argmax(score)
        label = labels.get(predicted_class_idx, "Unknown")
        confidence = float(score[predicted_class_idx])
        
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "spoiled_percentage": round(spoiled_percentage * 100, 2),
            "is_spoiled": spoiled_percentage > 0.5
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
