import cv2
import os
import numpy as np

KNOWN_DIR = "known_faces"
TRAINED_MODEL = "trained_model.yml"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

for filename in os.listdir(KNOWN_DIR):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(KNOWN_DIR, filename)
        label = os.path.splitext(filename)[0]
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_rects:
            faces.append(gray[y:y + h, x:x + w])
            labels.append(current_id)
            label_map[current_id] = label
        current_id += 1

# Save label map
np.save("labels.npy", label_map)

# Train and save model
recognizer.train(faces, np.array(labels))
recognizer.save(TRAINED_MODEL)
print("✅ Model trained and saved.")
