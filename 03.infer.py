from ultralytics import YOLO
import cv2

# load model
model = YOLO("runs/detect/train8/weights/best.pt")

# choose your source
# 0 = default webcam
# or replace with a file: "video.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open video capture")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run inference
    results = model(frame, verbose=False)

    # draw results on the frame
    annotated_frame = results[0].plot()

    # show it
    cv2.imshow("YOLOv8 Live", annotated_frame)

    # quit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
