import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === CONFIG ===
IMG_SIZE = (112, 112)
BATCH_SIZE = 64
VERIF_DIR = "../data/verification_data"
MODEL_PATH = "../models/classification_embed.h5"
OUTPUT_CSV = "../output/embeddings_classification.csv"

# === Enable GPU (Apple Metal) ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("[INFO] GPU memory growth enabled.")
else:
    print("[WARNING] No GPU found. Using CPU.")

# === Load Trained Model ===
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)
embedding_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer("embedding").output)

# === Prepare Image List and DataFrame ===
image_files = sorted([f for f in os.listdir(VERIF_DIR) if f.endswith(".jpg")])
df = pd.DataFrame({'filename': image_files})

# === Data Generator ===
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=VERIF_DIR,
    x_col='filename',
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

# === Batch Embedding Extraction ===
print(f"[INFO] Extracting embeddings for {len(df)} images...")
embeddings = embedding_model.predict(generator, verbose=1)

# === Debugging and Logging ===
print(f"[INFO] Embedding shape: {embeddings.shape}")  # Should be (num_images, 128)
print(f"[INFO] Example: {df['filename'].iloc[0]} → {embeddings[0][:5]}...")

# === Save to CSV ===
output = pd.DataFrame(embeddings)
output.insert(0, "filename", df['filename'])
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
output.to_csv(OUTPUT_CSV, index=False, header=False)

print(f"[✅ DONE] Saved embeddings to: {OUTPUT_CSV}")
