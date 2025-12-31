# config.py
import os

# Dataset & Paths
DATASET_YAML = "datasets/hardhat-or-hat/data.yaml"  # Path to your roboflow data.yaml
MODEL_VARIANT = "yolo11n.pt"  # Use 'n' for best Edge TPU performance
EXPORTED_MODEL_PATH = "runs/detect/train/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite"
TRAINED_MODEL = "runs/detect/train6/weights/best.pt"

# Training Params
DEVICE = 0  # 0 for GPU, "cpu" for CPU
EPOCHS = 10
IMG_SIZE = 512  # Standard for YOLO11
BATCH_SIZE = 32
SAVE_PERIOD = 1  # Save model every n epochs

# Inference Params
VIDEO_SOURCE = 0  # 0 for webcam
CONF_THRESHOLD = 0.25