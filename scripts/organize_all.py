import os
import sys
import glob

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
raw_dir = os.path.join(base_dir, 'raw')
org_dir = os.path.join(base_dir, 'organized')

categories = ['produce', 'bread', 'meat', 'fish', 'dairy', 'general', 'review']
states = ['fresh', 'mid', 'spoiled']

# Create directories
for cat in categories:
    for state in states:
        os.makedirs(os.path.join(org_dir, cat, state), exist_ok=True)

counts = {cat: {state: 0 for state in states} for cat in categories}

def determine_mapping(folder_name):
    cat = 'review'
    state = 'review'
    folder_name = folder_name.lower()
    
    # Category
    if 'fruit' in folder_name or 'vegetable' in folder_name or 'produce' in folder_name:
        cat = 'produce'
    elif 'bread' in folder_name or 'bun' in folder_name:
        cat = 'bread'
    elif 'meat' in folder_name or 'beef' in folder_name or 'pork' in folder_name:
        cat = 'meat'
    elif 'fish' in folder_name or 'seafood' in folder_name:
        cat = 'fish'
    elif 'dairy' in folder_name or 'milk' in folder_name or 'cheese' in folder_name:
        cat = 'dairy'
    elif 'food' in folder_name or 'general' in folder_name:
        cat = 'general'
        
    # State
    if 'fresh' in folder_name:
        state = 'fresh'
    elif 'stale' in folder_name or 'mid' in folder_name or 'half' in folder_name:
        state = 'mid'
    elif 'spoil' in folder_name or 'rotten' in folder_name or 'mold' in folder_name or 'bad' in folder_name:
        state = 'spoiled'
        
    if cat == 'review' or state == 'review':
        # If ambiguous, putting the state as fresh dummy to keep uniform structure
        if state == 'review': state = 'fresh'
        return 'review', state
        
    return cat, state

# Create symlinks
for root, dirs, files in os.walk(raw_dir):
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    if not images:
        continue
    
    folder_name = os.path.basename(root)
    parent_folder_name = os.path.basename(os.path.dirname(root))
    
    # Try using both to determine
    combined_name = folder_name + " " + parent_folder_name
    cat, state = determine_mapping(combined_name)
    
    for img in images:
        src = os.path.join(root, img)
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
        # To avoid collisions, prefix filename with dataset and folder
        new_filename = f"{dataset_name.replace('-','_')}_{folder_name}_{img}"
        dst = os.path.join(org_dir, cat, state, new_filename)
        
        try:
            if not os.path.exists(dst):
                # Using hardlinks as symlinks on windows often require admin privileges
                try:
                    os.symlink(src, dst)
                except OSError:
                    os.link(src, dst) # fallback to hardlink
            counts[cat][state] += 1
        except Exception as e:
            print(f"Failed to link {src} -> {dst}: {e}")

print(f"{'Category':<15} {'State':<10} {'Count':<10} {'Flag'}")
print("-" * 50)
for cat in categories:
    for state in states:
        if counts[cat][state] > 0 or (cat != 'review'):
            count = counts[cat][state]
            flag = "< 1,000" if count < 1000 else ""
            print(f"{cat:<15} {state:<10} {count:<10} {flag}")
