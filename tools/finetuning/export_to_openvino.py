from ultralytics import YOLO

# Load your newly trained model
model = YOLO('runs/detect/train8/weights/best.pt')

# Export to OpenVINO format (up to 3x faster on Intel CPUs)
model.export(format='openvino')