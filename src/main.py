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

# Branding
API_NAME = "Waste2Taste AI API"
API_METHODOLOGY = (
    "Spoilage: High-performance MobileNetV2 architecture with dynamic category Fallback. "
    "Sentiment: Zero-shot DeBERTa-v3 Transformer logic for zero-training moderation signals."
)

app = FastAPI(
    title=API_NAME,
    description="Unified API for Food Spoilage Detection and Multi-Tag Sentiment Moderation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global systems
loaded_models = {}
loaded_labels = {}
sentiment_model = None

# Spoilage Defaults
DEFAULT_LABELS = {0: "Fresh", 1: "Spoiled"}

# Sentiment/Moderation Configuration
SENTIMENT_DEFINITIONS = {
    "gratitude":      "grateful and satisfied with the food",
    "disappointment": "disappointed with the food quality",
    "disgust":         "disgusted, food was rotten or a health hazard",
    "frustration":     "frustrated with the merchant or pickup experience",
    "excitement":      "excited about a great deal or surprising find",
    "urgency":         "anxious or urgent about food expiring soon"
}

LABEL_TO_ID = {v: k for k, v in SENTIMENT_DEFINITIONS.items()}
CANDIDATE_LABELS = list(SENTIMENT_DEFINITIONS.values())

def load_spoilage_models():
    """Load all trained spoilage models from /models."""
    global loaded_models, loaded_labels
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir): return
    try:
        import onnxruntime as ort
    except ImportError: return

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
                else: loaded_labels[cat] = DEFAULT_LABELS
            except Exception as e:
                logger.error(f"Load error {cat}: {e}")

def load_sentiment_classifier():
    """Load the lite sentiment classifier."""
    global sentiment_model
    try:
        from transformers import pipeline
        model_name = "cross-encoder/nli-deberta-v3-small"
        sentiment_model = pipeline("zero-shot-classification", model=model_name)
    except Exception as e:
        logger.error(f"Sentiment load error: {e}")

@app.on_event("startup")
async def startup_event():
    load_spoilage_models()
    load_sentiment_classifier()

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except: return None

@app.get("/", tags=["Info"])
async def index():
    """Service status and capability discovery."""
    return {
        "status": "online",
        "api": API_NAME,
        "methodology": API_METHODOLOGY,
        "spoilage_models": list(loaded_models.keys()),
        "sentiment_config": {
            "model": "DeBERTa-v3-Lite",
            "active_labels": SENTIMENT_DEFINITIONS
        },
        "docs": "/docs"
    }

@app.post("/predict", tags=["Analysis"])
async def predict(
    file: UploadFile = File(...), 
    type: str = Query("general", description="Category (bread, meat, dairy, fish, produce)")
):
    """Detect spoilage using the specified category model."""
    global loaded_models, loaded_labels
    req_type = type.lower()
    used_type = req_type if req_type in loaded_models else "general"
    
    if used_type not in loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded.")

    session = loaded_models[used_type]
    labels = loaded_labels.get(used_type, DEFAULT_LABELS)

    contents = await file.read()
    processed_image = preprocess_image(contents)
    if processed_image is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    predictions = session.run(None, {session.get_inputs()[0].name: processed_image})[0]
    score = predictions[0]
    idx = np.argmax(score)
    
    spoil_idx = 1 # Fallback
    for i, lbl in labels.items():
        if lbl.lower() in ['rotten', 'spoiled', 'bad']:
            spoil_idx = i
            break
            
    return {
        "prediction": labels.get(idx, "Unknown"),
        "confidence": round(float(score[idx]), 4),
        "spoiled_percentage": round(float(score[spoil_idx]) * 100, 2),
        "is_spoiled": float(score[spoil_idx]) > 0.5,
        "metadata": {"requested": req_type, "used": used_type}
    }

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment", tags=["Analysis"])
async def sentiment(request: SentimentRequest):
    """Multi-tagging sentiment analysis. Returns fired tags (>0.3 score)."""
    if not sentiment_model:
        raise HTTPException(status_code=503, detail="Model offline.")

    result = sentiment_model(request.text, candidate_labels=CANDIDATE_LABELS, multi_label=True)
    
    # Tag-based response (multiple tags supported)
    tags = {}
    for l, s in zip(result["labels"], result["scores"]):
        if s > 0.3:
            tags[LABEL_TO_ID[l]] = round(s, 4)
    
    return {
        "tags": tags,
        "neutral": len(tags) == 0,
        "text": request.text
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
