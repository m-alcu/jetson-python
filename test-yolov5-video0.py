import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can change the model size

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Get annotated image
    annotated_frame = results.render()[0]

    print(results.pandas())

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
