from ultralytics import YOLO

# load pretrained YOLOv8 nano (or small)
model = YOLO("models/yolov8n.pt")

model.train(
    data="yolo_dataset/data.yaml",
    epochs=50,
    batch=32,
    imgsz=320, #640
    device="cpu",    # change to "0" if needed
    save_period=1
    #time=0.08333
)

print("Training complete.")