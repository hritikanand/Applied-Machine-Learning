import cv2
import face_recognition
import os

# Folder setup
KNOWN_FACES_DIR = "known_faces"
CAPTURED_DIR = "captured"
os.makedirs(CAPTURED_DIR, exist_ok=True)

# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"⚠️ No face found in {filename}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible. Please enable access in system settings.")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")
captured_count = 0
MAX_CAPTURES = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Resize for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]

    # Face detection
    face_locations = face_recognition.face_locations(rgb_small)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    else:
        face_encodings = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if matches and any(matches):
            best_match_index = face_distances.argmin()
            name = known_names[best_match_index]

        # Scale back to full frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        # Save up to 3 captures
        if name != "Unknown" and captured_count < MAX_CAPTURES:
            img_path = os.path.join(CAPTURED_DIR, f"{name}_{captured_count + 1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"📸 Saved: {img_path}")
            captured_count += 1

    # Show output
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or captured_count >= MAX_CAPTURES:
        break

cap.release()
cv2.destroyAllWindows()
