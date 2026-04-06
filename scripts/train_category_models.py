import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'organized')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Can be increased for better performance

os.makedirs(MODELS_DIR, exist_ok=True)

def train_category(category):
    cat_dir = os.path.join(DATA_DIR, category)
    if not os.path.exists(cat_dir):
        print(f"Directory {cat_dir} not found. Skipping {category}.")
        return

    # Check if we have enough states
    states = [s for s in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, s))]
    valid_states = []
    for state in states:
        imgs = [f for f in os.listdir(os.path.join(cat_dir, state)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(imgs) > 0:
            valid_states.append(state)
    
    if len(valid_states) < 2:
        print(f"Not enough data for {category} (found states: {valid_states}). Skipping.")
        return

    print(f"\nTraining model for category: {category}...")
    
    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255 * 2 - 1, # MobileNetV2 expects [-1, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        cat_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        cat_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save class indices
    class_indices = train_generator.class_indices
    with open(os.path.join(MODELS_DIR, f'labels_{category}.json'), 'w') as f:
        json.dump(class_indices, f)

    num_classes = len(class_indices)

    # Base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_all_entropy', metrics=['accuracy'])
    # Actually, categorical_crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint_path = os.path.join(MODELS_DIR, f'spoilage_{category}.keras')
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True)
    ]

    # Train
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    print(f"Finished training {category}. Model saved to {checkpoint_path}.")

if __name__ == "__main__":
    categories = ['bread', 'meat', 'dairy', 'fish', 'produce', 'general']
    for cat in categories:
        train_category(cat)
