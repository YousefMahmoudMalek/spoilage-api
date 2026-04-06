# Food Rescue AI Platform

This project provides a unified AI API for food spoilage detection and sentiment analysis for food rescue platforms.

## Features

- **Multi-Model Spoilage Detection**: Supports specialized models for different food categories (Bread, Meat, Dairy, Fish, etc.) with automatic fallback to a general model.
- **Sentiment Analysis**: Zero-shot classification tailored for food rescue interactions (satisfaction, disappointment, urgency, gratitude, frustration, excitement).
- **Railway Ready**: Pre-configured with a `Procfile` and resource-efficient dependencies for cloud deployment.

## Prerequisites

- **Python 3.9 - 3.11** (TensorFlow compatibility).
- At least 2GB of RAM (for sentiment analysis model).

## Setup

1.  **Virtual Environment**:
    ```bash
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## API Usage

### 1. Spoilage Detection
**Endpoint**: `POST /predict?model_type=general`
- **Body**: File (Image)
- **Model Types**: `general`, `bread`, `meat`, `dairy`, `fish`, `produce`.
- **Logic**: If the specific category model is not found, the `general` model is used automatically.

### 2. Sentiment Analysis
**Endpoint**: `POST /sentiment`
- **Body**: `{"text": "Your review here"}`
- **Emotions**:
  - `satisfaction`: Post-rescue positive outcome.
  - `disappointment`: Post-rescue negative outcome.
  - `urgency`: Rescue mindset, expiration pressure.
  - `gratitude`: Appreciation for saving food.
  - `frustration`: Process/app friction.
  - `excitement`: Discovery and deal joy.

### 3. API Documentation
Once running, visit `http://localhost:8000/docs` for interactive Swagger documentation.

## Deployment

This repository is ready for **Railway.app**:
1. Connect your GitHub repository.
2. Railway will automatically detect the `Procfile` and install dependencies.
3. Ensure you have the `models/` directory with at least `spoilage_model.keras` (tracked) for basic functionality.

## Project Structure

- `src/main.py`: FastAPI application with dynamic model loading.
- `models/`: Stores `.keras` models and `.json` label indices.
- `scripts/`: Data pipeline and training utilities.
- `data/`: (Ignored) Raw and organized datasets.
