from deepface import DeepFace
import sys

def detect_emotion(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        score = result[0]['emotion'][emotion]
        return emotion, score
    except Exception as e:
        return "Error", str(e)

# === Example Usage ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python emotion_detection.py path_to_image.jpg")
        sys.exit(1)

    path = sys.argv[1]
    emotion, score = detect_emotion(path)
    print(f"[🧠 EMOTION] {emotion.upper()} ({score:.2f} confidence)" if emotion != "Error" else f"[❌ ERROR] {score}")
