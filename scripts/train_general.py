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
DATA_DIR = os.path.join(BASE_DIR, 'data', 'organized', 'review')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1 

os.makedirs(MODELS_DIR, exist_ok=True)

def train_general():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    print("\nTraining General model from Review subset...")
    
    # Data preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255 * 2 - 1,
        validation_split=0.2
    )

    # We use a limited number of images from the 71k pool to keep it fast
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    class_indices = train_generator.class_indices
    with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)

    num_classes = len(class_indices)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = os.path.join(MODELS_DIR, 'spoilage_model.keras')
    model.fit(
        train_generator,
        steps_per_epoch=20, # Limit to 640 images per epoch for speed
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=10
    )
    model.save(checkpoint_path)
    print(f"Finished training General model. Saved to {checkpoint_path}.")

if __name__ == "__main__":
    train_general()
