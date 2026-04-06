import os
import sys
import shutil
import json
import urllib.request
import zipfile
import subprocess

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
local_kaggle_json = "kaggle.json"
user_kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

if os.path.exists(local_kaggle_json) and not os.path.exists(user_kaggle_json):
    shutil.copy(local_kaggle_json, user_kaggle_json)
    
if os.name != 'nt':
    try:
        os.chmod(user_kaggle_json, 0o600)
    except:
        pass

kaggle_datasets = [
    "maheen00shahid/fresh-and-spoiled-food-image-dataset",
    "ulnnproject/food-freshness-dataset",
    "sriramr/fruits-fresh-and-rotten-for-classification",
    "raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables",
    "swoyam2609/fresh-and-stale-classification",
    "muhammadaburayan/fish-freshness-classification",
    "vinayakshanawad/meat-freshness-image-dataset",
    "teranekerimova/moldy-bread-image-dataset",
    "nitinyadav4321/spoiled-food",
    "filipemonteir/fresh-and-rotten-fruits-and-vegetables"
]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
os.makedirs(base_dir, exist_ok=True)

def is_non_empty_dir(path):
    if not os.path.exists(path):
        return False
    if not os.path.isdir(path):
        return False
    # Check if there are any non-zip files
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.endswith('.zip'):
                return True
    return False

for dataset in kaggle_datasets:
    dataset_name = dataset.split('/')[-1]
    target_dir = os.path.join(base_dir, dataset_name)
    
    if is_non_empty_dir(target_dir):
        print(f"Skipping {dataset_name}, already exists and is non-empty.")
    else:
        print(f"\n=======================================================")
        print(f"Downloading Kaggle dataset: {dataset}...")
        print(f"=======================================================\n")
        os.makedirs(target_dir, exist_ok=True)
        # Using CLI as it has built-in retry logic, chunking, and progress bars
        kaggle_cmd = "kaggle"
        if os.name == 'nt':
            # Check if we are in a venv and using that python
            venv_bin = os.path.dirname(sys.executable)
            kaggle_exe = os.path.join(venv_bin, "kaggle.exe")
            if os.path.exists(kaggle_exe):
                kaggle_cmd = kaggle_exe
                
        subprocess.run([kaggle_cmd, "datasets", "download", "-d", dataset, "-p", target_dir, "--unzip"], check=False)

# Download Zenodo dataset using python package zenodo_get
zenodo_record_id = "13628374"
zenodo_dir = os.path.join(base_dir, "zenodo_meatscan")

if is_non_empty_dir(zenodo_dir):
    print("Skipping Zenodo MeatScan, already exists and is non-empty.")
else:
    print(f"\n=======================================================")
    print(f"Downloading Zenodo Record {zenodo_record_id}...")
    print(f"=======================================================\n")
    os.makedirs(zenodo_dir, exist_ok=True)
    subprocess.run([sys.executable, "-m", "zenodo_get", zenodo_record_id], cwd=zenodo_dir, check=False)
    
    # unzip any zip files downloaded
    for f in os.listdir(zenodo_dir):
        if f.endswith(".zip"):
            zip_path = os.path.join(zenodo_dir, f)
            print(f"Unzipping {f}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(zenodo_dir)
                os.remove(zip_path)
            except Exception as e:
                print(f"Failed to unzip {f}: {e}")

print("\nAll downloads complete!")
