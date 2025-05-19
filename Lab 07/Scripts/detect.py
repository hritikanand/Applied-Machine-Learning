from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Set path to your image
image_path = "/Users/hritikanand/Library/CloudStorage/OneDrive-SwinburneUniversity/Applied Machine Learning/Lab 07/objects.jpg"

# Load YOLOv8 model 
model = YOLO('yolov8n.pt')  

# Perform object detection
results = model(image_path)

# Save result with bounding boxes
output_path = os.path.join(os.path.dirname(image_path), "detected_objects.jpg")
results[0].save(filename=output_path)

# Display result
img = cv2.imread(output_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title("YOLOv8 Detected Objects")
plt.show()

print(f"\n Detection completed. Output saved to: {output_path}")
