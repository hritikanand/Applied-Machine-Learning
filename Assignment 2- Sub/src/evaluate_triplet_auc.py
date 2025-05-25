import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === CONFIG ===
IMG_SIZE = (112, 112)
BATCH_SIZE = 64
VERIF_DIR = "../data/verification_data"
MODEL_PATH = "../models/triplet_embed.h5"
PAIRS_FILE = "../data/verification_pairs_val.txt"
EMBEDDINGS_CSV = "../output/embeddings_triplet.csv"
ROC_PLOT = "../output/roc_triplet.png"

# === Load model ===
print("[INFO] Loading embedding model...")
model = load_model(MODEL_PATH)

# === Prepare file list ===
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

# === Predict all embeddings in batch ===
print(f"[INFO] Extracting embeddings for {len(df)} images in batches...")
embeddings = model.predict(generator, verbose=1)

# === Save embeddings to CSV ===
output_df = pd.DataFrame(embeddings)
output_df.insert(0, 'filename', df['filename'])
output_df.to_csv(EMBEDDINGS_CSV, index=False, header=False)

# === Cosine Similarity Function ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Load Embeddings as Dictionary ===
embeddings_dict = dict(zip(df['filename'], embeddings))

# === Compute Similarities ===
print("[INFO] Reading pairs and computing cosine similarities...")
pairs = []
labels = []

with open(PAIRS_FILE, 'r') as f:
    for line in f:
        img1, img2, label = line.strip().split()
        vec1 = embeddings_dict.get(os.path.basename(img1))
        vec2 = embeddings_dict.get(os.path.basename(img2))
        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity(vec1, vec2)
            pairs.append(sim)
            labels.append(int(label))

# === ROC and AUC ===
fpr, tpr, _ = roc_curve(labels, pairs)
roc_auc = auc(fpr, tpr)

# === Plot ===
plt.figure()
plt.plot(fpr, tpr, color='green', lw=2, label='AUC = %0.4f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Triplet Model')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(ROC_PLOT)

print(f"[✅ DONE] AUC = {roc_auc:.4f}")
print(f"[📊 ROC curve saved to] {ROC_PLOT}")
