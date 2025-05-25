import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# === Paths ===
EMBEDDINGS_CSV = "../output/embeddings_classification.csv"
PAIRS_FILE = "../data/verification_pairs_val.txt"
ROC_PLOT_PATH = "../output/roc_classification.png"

# === Cosine Similarity Function ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Load Embeddings ===
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].values
    vectors = df.iloc[:, 1:].values
    return dict(zip(names, vectors))

print("[INFO] Loading embeddings...")
embeddings = load_embeddings(EMBEDDINGS_CSV)

# === Load Verification Pairs and Compute Similarities ===
pairs = []
labels = []

print("[INFO] Reading verification pairs...")
with open(PAIRS_FILE, 'r') as f:
    for line in f:
        img1, img2, label = line.strip().split()
        vec1 = embeddings.get(os.path.basename(img1))
        vec2 = embeddings.get(os.path.basename(img2))
        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity(vec1, vec2)
            pairs.append(sim)
            labels.append(int(label))

print(f"[INFO] Evaluating {len(pairs)} valid pairs...")

# === ROC and AUC Calculation ===
fpr, tpr, _ = roc_curve(labels, pairs)
roc_auc = auc(fpr, tpr)

# === Plot ROC ===
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.4f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Classification-Based Face Verification')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(ROC_PLOT_PATH)

# === Log Results ===
print(f"[✅ DONE] AUC Score = {roc_auc:.4f}")
print(f"[📊 ROC Curve saved to] {ROC_PLOT_PATH}")
