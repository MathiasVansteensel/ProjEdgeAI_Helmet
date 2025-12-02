from ultralytics import YOLO

# load pretrained YOLO nano (or small)
model = YOLO("models/yolo11s.pt")

model.train(
    data="Yolo11Dataset/data.yaml",
    epochs=50,
    batch=32,
    imgsz=320, #640
    device="cpu",    # change to "0" if needed
    save_period=1
    #time=0.08333
)

print("Training complete.")