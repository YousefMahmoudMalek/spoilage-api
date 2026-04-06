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

app = FastAPI(title="Food Rescue AI Platform API")

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

    # Load other category models if present
    for filename in os.listdir(models_dir):
        if filename.startswith('spoilage_') and filename.endswith('.keras') and filename != 'spoilage_model.keras':
            cat = filename.replace('spoilage_', '').replace('.keras', '')
            model_path = os.path.join(models_dir, filename)
            try:
                import tensorflow as tf
                loaded_models[cat] = tf.keras.models.load_model(model_path)
                logger.info(f"Category model '{cat}' loaded.")
                
                # Check for specific labels
                labels_path = os.path.join(models_dir, f'labels_{cat}.json')
                if os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        indices = json.load(f)
                        loaded_labels[cat] = {v: k.capitalize() for k, v in indices.items()}
                else:
                    loaded_labels[cat] = DEFAULT_LABELS
            except Exception as e:
                logger.error(f"Failed to load category model '{cat}': {e}")

def load_sentiment_classifier():
    """Load the zero-shot sentiment classifier."""
    global sentiment_model
    try:
        from transformers import pipeline
        # Using a smaller model for better performance on limited hardware (like Railway)
        model_name = "cross-encoder/nli-deberta-v3-small"
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
        "models_loaded": list(loaded_models.keys()),
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
            return {"prediction": "Fresh", "confidence": 0.88, "used_model": "mock", "note": "MOCK RESPONSE"}
        raise HTTPException(status_code=503, detail=f"No spoilage models loaded (Requested: {requested_type})")

    model = loaded_models[used_type]
    labels = loaded_labels.get(used_type, DEFAULT_LABELS)

    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    if processed_image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        predictions = model.predict(processed_image)
        score = predictions[0]
        
        # Find which index corresponds to 'Rotten' or 'Spoiled'
        spoil_idx = -1
        for idx, lbl in labels.items():
            if lbl.lower() in ['rotten', 'spoiled', 'bad', 'moldy']:
                spoil_idx = idx
                break
        
        if spoil_idx == -1:
            spoil_idx = 1 if len(score) > 1 else 0
            
        spoiled_percentage = float(score[spoil_idx])
        
        # Determine overall label based on argmax
        predicted_class_idx = np.argmax(score)
        label = labels.get(predicted_class_idx, "Unknown")
        confidence = float(score[predicted_class_idx])
        
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "spoiled_percentage": round(spoiled_percentage * 100, 2),
            "is_spoiled": spoiled_percentage > 0.5,
            "model_type_requested": requested_type,
            "model_type_used": used_type
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

    labels = ["satisfaction", "disappointment", "urgency", "gratitude", "frustration", "excitement"]
    
    try:
        result = sentiment_model(request.text, candidate_labels=labels)
        
        # Format results: list of {label: ..., score: ...} sorted by score
        scores = [{"label": label, "score": round(score, 4)} for label, score in zip(result["labels"], result["scores"])]
        
        return {
            "top_sentiment": result["labels"][0],
            "confidence": round(result["scores"][0], 4),
            "all_sentiments": scores,
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
