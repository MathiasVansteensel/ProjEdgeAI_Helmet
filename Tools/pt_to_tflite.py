from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./runs/detect/train5/weights/best.pt")
# Export the model to TFLite format

path = model.export(format="tflite")  # creates 'yolo11n_float32.tflite'
print("Exported to:", path)

# Load the exported TFLite model
tflite_model = YOLO("yolo11n_float32.tflite")

# Run inference
results = tflite_model("https://ultralytics.com/images/bus.jpg")