from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./runs/detect/train5/weights/best.pt")

# Export the model to TFLite Edge TPU format
path = model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'
print("Exported to:", path)

# Load the exported TFLite Edge TPU model
edgetpu_model = YOLO("yolo11n_full_integer_quant_edgetpu.tflite")

# Run inference
results = edgetpu_model("https://ultralytics.com/images/bus.jpg")