from ultralytics import YOLO
model = YOLO("./models/yolo8n_p2new/yolov8n_p2new_mblur_40_db.pt") # 載入 .pt
model.export(format="coreml", half=True, nms=True, imgsz=640) # 這會產生含 NMS 的完美模型
model.export(format="tflite", half=True, nms=True, imgsz=640, data = "./Detect Baseballs.v3-train-1280-720.yolov8/data.yaml") # 這會產生含 NMS 的完美模型