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

app = FastAPI(
    title="Food Rescue AI Platform",
    description="Unified API for Food Spoilage Detection and Moderation Sentiment Analysis.",
    version="1.2.0"
)

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
    """Load all trained spoilage models using ONNX Runtime for minimal footprint."""
    global loaded_models, loaded_labels
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found at {models_dir}")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed.")
        return

    for filename in os.listdir(models_dir):
        if filename.endswith('.onnx'):
            is_general = (filename == 'spoilage_model.onnx')
            cat = 'general' if is_general else filename.replace('spoilage_', '').replace('.onnx', '')
            model_path = os.path.join(models_dir, filename)
            try:
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                loaded_models[cat] = session
                
                label_filename = 'class_indices.json' if is_general else f'labels_{cat}.json'
                labels_path = os.path.join(models_dir, label_filename)
                if os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        indices = json.load(f)
                        loaded_labels[cat] = {v: k.capitalize() for k, v in indices.items()}
                else:
                    loaded_labels[cat] = DEFAULT_LABELS
            except Exception as e:
                logger.error(f"Failed to load ONNX model '{cat}': {e}")

def load_sentiment_classifier():
    """Load the zero-shot sentiment classifier."""
    global sentiment_model
    try:
        from transformers import pipeline
        # Switched to the Lite model (140MB) for faster Free Tier performance
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
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.get("/", tags=["General"])
async def index():
    """Returns platform status and list of available AI models."""
    return {
        "status": "online",
        "engine": "ONNX Runtime + DeBERTa-v3-Lite",
        "available_spoilage_models": list(loaded_models.keys()),
        "sentiment_ready": sentiment_model is not None,
        "docs": "/docs"
    }

@app.post("/predict", tags=["AI Prediction"])
async def predict(
    file: UploadFile = File(...), 
    model_type: str = Query(
        "general", 
        description="Type of food category (e.g., bread, meat, dairy, fish, produce). Defaults to 'general' if category not found."
    )
):
    """Detect spoilage/freshness. Use model_type to specify a category."""
    global loaded_models, loaded_labels
    
    requested_type = model_type.lower()
    used_type = requested_type if requested_type in loaded_models else "general"
    
    if used_type not in loaded_models:
        raise HTTPException(status_code=503, detail="No base spoilage model loaded.")

    session = loaded_models[used_type]
    labels = loaded_labels.get(used_type, DEFAULT_LABELS)

    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    if processed_image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: processed_image})[0]
        score = predictions[0]
        predicted_class_idx = np.argmax(score)
        label = labels.get(predicted_class_idx, "Unknown")
        confidence = float(score[predicted_class_idx])
        
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
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment", tags=["AI Prediction"])
async def sentiment(request: SentimentRequest):
    """Classify user interaction into moderation-centric sentiment categories."""
    global sentiment_model
    if not sentiment_model:
        raise HTTPException(status_code=503, detail="Sentiment model is not loaded.")

    try:
        result = sentiment_model(request.text, candidate_labels=CANDIDATE_LABELS, multi_label=True)
        qualified_sentiments = []
        for l, s in zip(result["labels"], result["scores"]):
            if s > 0.3:
                qualified_sentiments.append({
                    "id": LABEL_MAP.get(l, "unknown"),
                    "label": l,
                    "score": round(s, 4)
                })
        
        qualified_sentiments = sorted(qualified_sentiments, key=lambda x: x["score"], reverse=True)
        
        return {
            "labels": qualified_sentiments,
            "neutral": len(qualified_sentiments) == 0,
            "text": request.text
        }
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "ok", "spoilage_models": list(loaded_models.keys())}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
