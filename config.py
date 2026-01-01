# config.py
import os

# Dataset & Paths
DATASET_YAML = "datasets/hardhat-or-hat/data.yaml"  # Path to your roboflow data.yaml
MODEL_VARIANT = "yolo11s.pt"  # Use 'n' for best Edge TPU performance
EXPORTED_MODEL_PATH = "runs/detect/train6/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite"
TRAINED_MODEL = "runs/detect/train6/weights/best.pt"

# Training Params
DEVICE = 0  # 0 for GPU, "cpu" for CPU
EPOCHS = 600
IMG_SIZE = 512  # Standard for YOLO11
BATCH_SIZE = 64
SAVE_PERIOD = 5  # Save model every n epochs

# Inference Params
VIDEO_SOURCE = "E:\\Downloads\\busy-roof-construction-SBV-300154131-preview.mp4"  # 0 for webcam
CONF_THRESHOLD = 0.25