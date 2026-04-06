import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inspect_out.txt'))

with open(out_path, 'w') as f:
    f.write("Inspecting downloaded datasets:\n")
    if not os.path.exists(base_dir):
        f.write("Base directory does not exist.\n")
    else:
        for dataset in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue
            
            f.write(f"\nDataset: {dataset}\n")
            leaf_dirs = []
            for root, dirs, files in os.walk(dataset_path):
                img_count = sum(1 for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')))
                if img_count > 0:
                    rel_path = os.path.relpath(root, dataset_path)
                    leaf_dirs.append((rel_path, img_count))
            
            if not leaf_dirs:
                f.write("  No image directories found.\n")
            for rel_path, count in leaf_dirs:
                f.write(f"  {rel_path}: {count} images\n")
