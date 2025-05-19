# creative_art.py
from PIL import Image
import os
import shutil

# Create output folder if it doesn't exist
output_dir = "generated_outputs"
os.makedirs(output_dir, exist_ok=True)

# Local source image path (choose one)
source_path = "GAN images/1.png"  # or change to "GAN images/1000.png"
target_path = os.path.join(output_dir, "anime_face.png")

# Load and save
try:
    img = Image.open(source_path)
    img.save(target_path)
    print(f"✅ Successfully copied '{source_path}' to '{target_path}'")
except FileNotFoundError:
    print("❌ Source image not found. Check file name or path.")
