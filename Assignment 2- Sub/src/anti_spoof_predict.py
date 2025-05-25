import cv2
import numpy as np

def is_real_face(image_path):
    """
    Basic anti-spoof detection based on texture — NOT foolproof, but good for demo.
    Returns True if real, False if likely spoof.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Threshold for texture detail — tweak based on demo quality
    return laplacian_var > 100

# === Example Usage ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python anti_spoof_predict.py path_to_image.jpg")
        sys.exit(1)

    path = sys.argv[1]
    result = is_real_face(path)
    if result:
        print("[✅ REAL] Live face detected")
    else:
        print("[❌ SPOOF] Fake/spoofed image detected")
