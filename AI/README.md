# Food Rescue AI Platform

This project provides a unified AI API for food spoilage detection and sentiment analysis for food rescue platforms.

## Features

- **Multi-Model Spoilage Detection**: Supports specialized models for different food categories (Bread, Meat, Dairy, Fish, etc.) with automatic fallback to a general model.
- **Sentiment Analysis**: Zero-shot classification tailored for food rescue interactions (satisfaction, disappointment, urgency, gratitude, frustration, excitement).
- **Railway Ready**: Pre-configured with a `Procfile` and resource-efficient dependencies for cloud deployment.

## Setup & Training

The project is now organized into a self-contained `AI/` subdirectory for better portability.

1.  **Virtual Environment**:
    ```bash
    cd AI
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline**:
    We provide a controller script that handles the entire workflow step-by-step:
    ```bash
    python scripts/pipeline.py
    ```
    This will:
    - **01 Download**: Fetch datasets from Kaggle (requires `kaggle.json` in `AI/`).
    - **02 Organize**: Map raw data into categorized folders.
    - **03 Train**: Perform transfer learning using MobileNetV2 (est. 45-90 mins).
    - **04 Export**: Convert trained models to ONNX for fast inference.
    - **05 Evaluate**: Generate performance reports.

## Project Structure

```text
/ (root)
├── railway.json           # Railway deployment config
└── AI/                    # Self-contained AI module
    ├── scripts/           # Sequential pipeline (01-05) + pipeline.py
    ├── src/               # FastAPI application logic
    ├── models/            # Trained .h5 and .onnx models
    ├── data/              # Organized datasets (ignored by git)
    ├── dataset/           # Raw Kaggle downloads (ignored by git)
    ├── Dockerfile         # Container config for Railway
    └── requirements.txt   # Python dependencies
```

## Deployment

This project is configured for **Railway.app**:
1. Connect your GitHub repository.
2. Railway will detect `railway.json` in the root and use the `AI/Dockerfile` for deployment.
3. The context is set to the `AI/` folder, ensuring all internal paths remain consistent.
