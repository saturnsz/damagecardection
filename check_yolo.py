from ultralytics import YOLO

try:
    model = YOLO("best (1).pt")
    print("YOLO classes:", model.names)
except Exception as e:
    print(f"Error: {e}")
