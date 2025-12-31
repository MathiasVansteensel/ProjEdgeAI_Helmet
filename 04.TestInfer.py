# 04.TestInfer.py
import cv2
from ultralytics import YOLO
from config import VIDEO_SOURCE, EXPORTED_MODEL_PATH, CONF_THRESHOLD

def run_inference():
    # Load the exported Edge TPU model
    # Note: On a PC without a Coral USB stick, this may fall back to CPU
    model = YOLO(EXPORTED_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference
        results = model(frame, conf=CONF_THRESHOLD)

        # Visualize
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Hardhat Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()