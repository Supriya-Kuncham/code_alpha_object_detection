import torch
import cv2

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Optional: Set resolution (low res = faster detection)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed (optional but recommended)
    resized_frame = cv2.resize(frame, (640, 480))

    # Perform detection
    results = model(resized_frame)

    # Render the results
    annotated_frame = results.render()[0]

    # Show the frame
    cv2.imshow('YOLOv5 Detection', annotated_frame)

    # Wait for 'q' key to stop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
