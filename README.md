# Food Spoilage Detection API

This project provides a FastAPI server for detecting food spoilage from images, using a MobileNetV2 model.

## Prerequisites

- **Python 3.9 - 3.11** (Critical: TensorFlow does not yet support Python 3.12+ or 3.14).
- CUDA compatible GPU (optional, for faster training).
- Kaggle API credentials.

## Setup

1.  **Install Python 3.10**: Unfortunately, your current Python 3.14 environment is too new for the required Machine Learning libraries (TensorFlow). Please install Python 3.10 from [python.org](https://www.python.org/downloads/).

2.  **Create a Virtual Environment**:
    ```bash
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you are on standard Python 3.14, `tensorflow` installation will fail.*

4.  **Kaggle Credentials**:
    - Ensure `kaggle.json` is present in the root directory (it has been created with your provided key).
    - Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Usage

### 1. Data Pipeline
Download and organize the datasets:
```bash
python scripts/download_datasets.py
```
This will download "Fresh and Rotten Fruits" and other datasets to the `dataset/` folder.

### 2. Data Organization
Organize the downloaded data into "fresh" and "rotten" categories:
```bash
python scripts/organize_data.py
```

### 3. Training
Train the MobileNetV2 model:
```bash
python scripts/train.py
```
- The model will be saved to `models/spoilage_model.keras`.
- Class indices will be saved to `models/class_indices.json`.

### 3. API Server
Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```
- API Docs: `http://localhost:8000/docs`
- Endpoint: `POST /predict` (Upload an image)

## Project Structure
- `src/main.py`: FastAPI application.
- `scripts/train.py`: Training script (MobileNetV2).
- `scripts/download_datasets.py`: Data download script.
- `dataset/`: Stores raw and processed data.
- `models/`: Stores trained models.
