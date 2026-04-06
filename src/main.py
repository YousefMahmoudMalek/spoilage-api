from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
import os
import json
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Rescue Moderation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and labels variable
loaded_models = {}
loaded_labels = {}
sentiment_model = None

# Default labels mapping for spoilage
DEFAULT_LABELS = {0: "Fresh", 1: "Spoiled"}

# Sentiment specific configuration
CANDIDATE_LABELS = [
    "grateful and satisfied with the food",
    "disappointed with the food quality",
    "disgusted, food was rotten or a health hazard",
    "frustrated with the merchant or pickup experience",
    "excited about a great deal or surprising find",
    "anxious or urgent about food expiring soon"
]

LABEL_MAP = {
    "grateful and satisfied with the food":              "gratitude",
    "disappointed with the food quality":                "disappointment",
    "disgusted, food was rotten or a health hazard":     "disgust",
    "frustrated with the merchant or pickup experience": "frustration",
    "excited about a great deal or surprising find":     "excitement",
    "anxious or urgent about food expiring soon":        "urgency"
}

def load_spoilage_models():
    """Load all trained spoilage models from the models directory."""
    global loaded_models, loaded_labels
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found at {models_dir}")
        return

    # Try loading the general model first
    general_path = os.path.join(models_dir, 'spoilage_model.keras')
    if os.path.exists(general_path):
        try:
            import tensorflow as tf
            # Load without GPU to save memory if needed
            loaded_models['general'] = tf.keras.models.load_model(general_path)
            logger.info("General spoilage model loaded.")
            
            # Load associated labels if they exist
            labels_path = os.path.join(models_dir, 'class_indices.json')
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    indices = json.load(f)
                    loaded_labels['general'] = {v: k.capitalize() for k, v in indices.items()}
            else:
                loaded_labels['general'] = DEFAULT_LABELS
        except Exception as e:
            logger.error(f"Failed to load general model: {e}")

def load_sentiment_classifier():
    """Load the zero-shot sentiment classifier."""
    global sentiment_model
    try:
        from transformers import pipeline
        # facebook/bart-large-mnli is excellent for zero-shot but large (~1.6GB)
        model_name = "facebook/bart-large-mnli"
        sentiment_model = pipeline("zero-shot-classification", model=model_name)
        logger.info(f"Sentiment classifier loaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")

@app.on_event("startup")
async def startup_event():
    load_spoilage_models()
    load_sentiment_classifier()

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

@app.get("/")
async def root():
    return {
        "status": "online",
        "models": list(loaded_models.keys()),
        "sentiment_ready": sentiment_model is not None,
        "docs": "/docs"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    model_type: str = Query("general", description="Type of spoilage model to use")
):
    global loaded_models, loaded_labels
    
    # Fallback logic
    requested_type = model_type.lower()
    used_type = requested_type if requested_type in loaded_models else "general"
    
    if used_type not in loaded_models:
        if os.environ.get("MOCK_MODE") == "true":
            return {"prediction": "Fresh", "confidence": 0.88, "used_model": "mock"}
        raise HTTPException(status_code=503, detail=f"No models loaded (Requested: {requested_type})")

    model = loaded_models[used_type]
    labels = loaded_labels.get(used_type, DEFAULT_LABELS)

    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    if processed_image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        predictions = model.predict(processed_image)
        score = predictions[0]
        
        # Determine overall label based on argmax
        predicted_class_idx = np.argmax(score)
        label = labels.get(predicted_class_idx, "Unknown")
        confidence = float(score[predicted_class_idx])
        
        # Find which index corresponds to 'Rotten' for spoiled percentage mapping
        spoil_idx = -1
        for idx, lbl in labels.items():
            if lbl.lower() in ['rotten', 'spoiled', 'bad']:
                spoil_idx = idx
                break
        
        if spoil_idx == -1:
            spoil_idx = 1 if len(score) > 1 else 0
            
        spoiled_percentage = float(score[spoil_idx])
        
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "spoiled_percentage": round(spoiled_percentage * 100, 2),
            "is_spoiled": spoiled_percentage > 0.5,
            "metadata": {
                "model_requested": requested_type,
                "model_used": used_type,
                "fallback_active": requested_type != used_type
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
async def sentiment(request: SentimentRequest):
    global sentiment_model
    if not sentiment_model:
        raise HTTPException(status_code=503, detail="Sentiment model is not loaded.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

    try:
        # Multi-label classification with threshold logic
        result = sentiment_model(
            request.text, 
            candidate_labels=CANDIDATE_LABELS, 
            multi_label=True
        )
        
        # Filter and map labels above 0.3 threshold
        qualified_sentiments = []
        for label, score in zip(result["labels"], result["scores"]):
            if score > 0.3:
                mapped_id = LABEL_MAP.get(label, "unknown")
                qualified_sentiments.append({
                    "id": mapped_id,
                    "label": label,
                    "score": round(score, 4)
                })
        
        # Sort by score descending (already sorted by bart by default, but to be safe)
        qualified_sentiments = sorted(qualified_sentiments, key=lambda x: x["score"], reverse=True)
        
        return {
            "labels": qualified_sentiments,
            "neutral": len(qualified_sentiments) == 0,
            "text": request.text
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "spoilage_models": list(loaded_models.keys()),
        "sentiment_loaded": sentiment_model is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
