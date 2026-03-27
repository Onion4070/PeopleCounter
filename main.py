import cv2
from ultralytics import YOLO

model = YOLO(model='yolov8n.pt')
cap = cv2.VideoCapture(0)
print('program ready')

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()