import os
import uuid

def shuffle_filenames():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data', 'organized_new')
    
    print("Randomizing filenames to fix Keras validation split bias...")
    renamed_count = 0
    
    for cat in os.listdir(data_dir):
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
            
        for state in os.listdir(cat_dir):
            state_dir = os.path.join(cat_dir, state)
            if not os.path.isdir(state_dir):
                continue
                
            for filename in os.listdir(state_dir):
                # Skip if already randomized with a uuid (length check)
                if len(filename) > 32 and '-' in filename[:36]:
                    continue
                    
                old_path = os.path.join(state_dir, filename)
                
                # Prepend a random 8-character string to ensure alphabetical sorting is completely random
                random_prefix = str(uuid.uuid4())[:8]
                new_filename = f"{random_prefix}_{filename}"
                new_path = os.path.join(state_dir, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")
                    
    print(f"Successfully randomized {renamed_count} files!")
    print("Your validation splits will now be truly representative.")

if __name__ == "__main__":
    shuffle_filenames()
