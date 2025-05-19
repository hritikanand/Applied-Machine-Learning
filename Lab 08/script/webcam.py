import cv2
import os
import numpy as np

# Load recognizer & label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CAPTURED_DIR = "captured"
os.makedirs(CAPTURED_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")
captured = 0
MAX_CAPTURES = 3

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi)

        name = label_map.get(id_, "Unknown")
        text = f"{name} ({round(confidence, 2)})" if confidence < 100 else "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if name != "Unknown" and captured < MAX_CAPTURES:
            path = os.path.join(CAPTURED_DIR, f"{name}_{captured + 1}.jpg")
            cv2.imwrite(path, frame)
            print(f"📸 Saved: {path}")
            captured += 1

    cv2.imshow("OpenCV Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or captured >= MAX_CAPTURES:
        break

cap.release()
cv2.destroyAllWindows()
