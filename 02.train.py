# 02.Train.py
from ultralytics import YOLO
from config import MODEL_VARIANT, DATASET_YAML, EPOCHS, IMG_SIZE, BATCH_SIZE, DEVICE, SAVE_PERIOD

def main():
    # Load a pretrained model
    model = YOLO(MODEL_VARIANT)

    # Train the model
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        save_period=SAVE_PERIOD,
        plots=True
    )

if __name__ == "__main__":
    main()