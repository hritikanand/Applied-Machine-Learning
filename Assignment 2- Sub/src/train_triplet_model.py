import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam

# === CONFIG ===
DATA_DIR = "../data/classification_data/train_data"
IMG_SIZE = (112, 112)
BATCH_SIZE = 32
EPOCHS = 10
MARGIN = 1.0
MODEL_PATH = "../models/triplet_embed.h5"

# === GPU Enablement ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("[INFO] GPU memory growth enabled.")

# === Triplet Loss Function ===
def triplet_loss(y_true, y_pred):
    anchor, positive, negative = tf.split(y_pred, 3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + MARGIN, 0.0)
    return tf.reduce_mean(loss)

# === Define Embedding Model ===
def create_embedding_model():
    inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = Conv2D(32, (3, 3), activation='relu')(inp)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    model = Model(inputs=inp, outputs=x, name='embedding_model')
    return model

embedding_model = create_embedding_model()

# === Build Triplet Network ===
anchor_in = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
positive_in = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
negative_in = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

anchor_emb = embedding_model(anchor_in)
positive_emb = embedding_model(positive_in)
negative_emb = embedding_model(negative_in)

merged_output = Lambda(lambda x: tf.concat(x, axis=1))([anchor_emb, positive_emb, negative_emb])
triplet_model = Model([anchor_in, positive_in, negative_in], merged_output)
triplet_model.compile(optimizer=Adam(1e-4), loss=triplet_loss)

# === Helper: Preprocess Image ===
def preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    return img_to_array(img) / 255.0

# === Generate Valid Triplet Batch ===
def create_triplet_batch(batch_size):
    classes = [c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))]
    triplets = []

    while len(triplets) < batch_size:
        anchor_class = random.choice(classes)
        positive_class = anchor_class
        negative_class = random.choice([c for c in classes if c != anchor_class])

        anchor_imgs = os.listdir(os.path.join(DATA_DIR, anchor_class))
        negative_imgs = os.listdir(os.path.join(DATA_DIR, negative_class))

        if len(anchor_imgs) < 2 or len(negative_imgs) == 0:
            continue

        anchor_img = random.choice(anchor_imgs)
        positive_img = random.choice([img for img in anchor_imgs if img != anchor_img])
        negative_img = random.choice(negative_imgs)

        anchor = preprocess_image(os.path.join(DATA_DIR, anchor_class, anchor_img))
        positive = preprocess_image(os.path.join(DATA_DIR, positive_class, positive_img))
        negative = preprocess_image(os.path.join(DATA_DIR, negative_class, negative_img))

        triplets.append((anchor, positive, negative))

    anchors = np.array([t[0] for t in triplets])
    positives = np.array([t[1] for t in triplets])
    negatives = np.array([t[2] for t in triplets])
    return [anchors, positives, negatives], np.zeros((batch_size,))

# === Training Loop ===
print("[INFO] Starting triplet training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    triplet_inputs, dummy_labels = create_triplet_batch(BATCH_SIZE * 5)
    loss = triplet_model.train_on_batch(triplet_inputs, dummy_labels)
    print(f"  ➤ Batch Loss: {loss:.4f}")

# === Save Only the Embedding Model ===
embedding_model.save(MODEL_PATH)
print(f"[✅ DONE] Triplet embedding model saved to {MODEL_PATH}")
