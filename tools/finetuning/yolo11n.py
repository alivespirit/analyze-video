from ultralytics import YOLO

# Load the standard "Nano" model (it will auto-download)
# If this is too inaccurate, change 'yolov8n.pt' to 'yolov8s.pt' (Small - more accurate, slightly slower)
model = YOLO('yolo11n.pt')

# Run inference on your video
# conf=0.4 means it ignores low-confidence detections
# save=True will create a video file showing what it detected
model.predict(source="09M02S_1765357742.mp4", show=True, conf=0.4, save=True)