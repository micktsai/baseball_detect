from ultralytics import YOLO

model = YOLO(
    "runs/detect/my_baseball_run/weights/best.pt"
)

model.export(
    format="tflite",
    half=True,
    imgsz=640
)