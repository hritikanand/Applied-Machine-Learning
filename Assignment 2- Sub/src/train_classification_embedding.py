import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

# Paths
TRAIN_DIR = "../data/classification_data/train_data"
MODEL_PATH = "../models/classification_embed.keras"
LABEL_MAP_PATH = "../models/label_map.json"

# Image settings
IMG_SIZE = (112, 112)
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# Build model
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu', name='embedding'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, epochs=EPOCHS)

# Save model in .keras format
model.save(MODEL_PATH)

# Save label map
label_map = train_gen.class_indices
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)

print(f"[✅] Model saved to {MODEL_PATH}")
