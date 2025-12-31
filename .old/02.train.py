from ultralytics import YOLO

if __name__ == '__main__':
    # load pretrained YOLO nano (or small)
    model = YOLO("models/yolo11s.pt")

    model.train(
        data="Yolo11Dataset/data.yaml",
        epochs=250,
        batch=64,
        imgsz=512, #640
        device="0",    # change to "0" if needed
        save_period=1
        #time=0.08333
    )

    print("Training complete.")