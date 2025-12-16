from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # load model
    model = YOLO("C:/Users/mathi/Downloads/best.pt")

    # choose your source
    # 0 = default webcam
    # or replace with a file: "video.mp4"
    cap = cv2.VideoCapture("C:/Users/mathi/Downloads/ConstructionAhhVideo.mp4")

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
        cv2.imshow("YOLO Live", annotated_frame)

        # quit on ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()