from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8_fine_tuned.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("Office CCTV.mp4", save=True, imgsz=640, conf=0.5, iou=0.8, stream_buffer=True)