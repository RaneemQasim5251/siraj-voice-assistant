from ultralytics import YOLO

# Create models directory if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Download YOLOv8n model
model = YOLO('yolov8n.pt')

# Save the model
model.save('models/yolov8n.pt') 