import os
import shutil
import glob

# Configuration
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'raw_fruits')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'processed')

def organize():
    if not os.path.exists(RAW_DIR):
        print(f"Raw dataset not found at {RAW_DIR}. Please run download_datasets.py first.")
        return

    # Create target directories
    for category in ['fresh', 'rotten']:
        os.makedirs(os.path.join(PROCESSED_DIR, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, 'validation', category), exist_ok=True)

    print(f"Scanning {RAW_DIR}...")
    
    # We expect a structure like: raw_fruits/dataset/train/freshapples/...
    # or just folders. We will walk through everything.
    
    file_count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()
                
                # Determine category
                category = None
                if 'fresh' in folder_name:
                    category = 'fresh'
                elif 'rotten' in folder_name or 'stale' in folder_name:
                    category = 'rotten'
                
                if category:
                    # Determine split (train/test/validation) from path if possible
                    # otherwise default to train (and let ImageDataGenerator split it)
                    # But simpler: put all in 'train' and let Keras split validation
                    target_dir = os.path.join(PROCESSED_DIR, 'train', category)
                    
                    # Copy file
                    try:
                        shutil.copy2(file_path, target_dir)
                        file_count += 1
                        if file_count % 1000 == 0:
                            print(f"Processed {file_count} images...")
                    except Exception as e:
                        print(f"Error copying {file}: {e}")

    print(f"Done. Organized {file_count} images into {PROCESSED_DIR}")

if __name__ == "__main__":
    organize()
