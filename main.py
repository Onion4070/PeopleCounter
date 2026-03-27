import cv2
from ultralytics import YOLO

model = YOLO(model='yolov8n.pt')
cap = cv2.VideoCapture(0)
print('program ready')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    if results[0].boxes == None:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id

    if ids == None:
        continue

    ids = ids.cpu().numpy().astype(int)

    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    scores = results[0].boxes.conf.cpu().numpy()

    for box, track_id, cls, score in zip(boxes, ids, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = results[0].names[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, str(score), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()