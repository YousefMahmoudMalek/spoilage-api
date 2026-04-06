import os
import sys

kaggle_dir = os.path.expanduser("~/.kaggle")
print("Kaggle directory:", kaggle_dir)
user_kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
print("Kaggle JSON path:", user_kaggle_json)
print("Exists:", os.path.exists(user_kaggle_json))

if os.path.exists(user_kaggle_json):
    with open(user_kaggle_json, 'r') as f:
        print("Kaggle JSON content:", f.read()[:20] + "...")

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Authentication successful!")
    print("Testing download of a small dataset...")
    # Just list files in maheen00shahid/fresh-and-spoiled-food-image-dataset
    files = api.dataset_list_files('maheen00shahid/fresh-and-spoiled-food-image-dataset')
    print("Files found:", len(files.files) if hasattr(files, 'files') else "None")
except Exception as e:
    print("Error:", e)
