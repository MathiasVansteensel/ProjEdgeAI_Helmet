######################
# TO BE RUN ON LINUX #
######################

# 03.Convert.py
from ultralytics import YOLO
from config import IMG_SIZE, TRAINED_MODEL
import os

# Load your trained model (usually in runs/detect/train/weights/best.pt)

def export_for_coral():
    model = YOLO(TRAINED_MODEL)
    
    print("Exporting to TFLite (Edge TPU compatible)...")
    # format='edgetpu' handles the INT8 quantization automatically
    model.export(format="edgetpu", imgsz=IMG_SIZE)
    
    print("Export complete. Look for 'best_full_integer_quant_edgetpu.tflite' in your weights folder.")

if __name__ == "__main__":
    export_for_coral()