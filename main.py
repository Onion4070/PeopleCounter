import cv2
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        if isinstance(other, tuple):
            other = Point(*other)
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if isinstance(other, tuple):
            other = Point(*other)
        return Point(self.x - other.x, self.y - other.y)

# tupleをPointクラスに変換
def _to_point(p):
    return p if isinstance(p, Point) else Point(*p)

# 外積(2D)
def cross2d(v1: Point, v2: Point):
    v1 = _to_point(v1)
    v2 = _to_point(v2)
    return v1.x * v2.y - v1.y * v2.x

# 線分交差判定
def intersect(v1: Point, v2: Point, v3: Point, v4: Point):
    v1v2 = v2 - v1
    v1v3 = v3 - v1
    v1v4 = v4 - v1

    v3v1 = v1 - v3
    v3v2 = v2 - v3
    v3v4 = v4 - v3

    # v1v2から見てv3, v4が別領域にあれば交差
    # 同様にv3v4から見てv1, v2が別領域にあれば交差
    # 接触も許容する
    return (cross2d(v1v2, v1v3) * cross2d(v1v2, v1v4) <= 0
        and cross2d(v3v4, v3v1) * cross2d(v3v4, v3v2) <= 0)

model = YOLO(model='yolov8s.pt')
print('Model loaded')
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video2.mp4')
WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 判定するライン
line_x1 = WIDTH // 2 - 320
line_x2 = WIDTH // 2 + 320
line_y1 = HEIGHT // 2
line_y2 = HEIGHT // 2

line_p1 = Point(line_x1, line_y1)
line_p2 = Point(line_x2, line_y2)

track_history = {}
count_in = 0
count_out = 0

counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, 
                          classes=[0], 
                          conf=0.5,
                          persist=True)
    if results[0].boxes is None:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id

    if ids == None:
        continue

    ids = ids.cpu().numpy().astype(int)

    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    scores = results[0].boxes.conf.cpu().numpy()

    for box, track_id, cls, score in zip(boxes, ids, classes, scores):
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box)
        label = results[0].names[cls]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 中心点描画
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 255), -1)

        # 物体領域描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ラベル/確信度描画
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, str(score), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if track_id in track_history:
            prev_pt = track_history[track_id]
            curr_pt = Point(cx, cy)

            # 線分交差判定
            if intersect(prev_pt, curr_pt, line_p1, line_p2):
                if track_id not in counted_ids:
                    # 方向検出
                    direction = cross2d(line_p2 - line_p1, curr_pt - prev_pt)

                    if direction > 0:
                        count_in += 1
                    else:
                        count_out += 1
                    counted_ids.add(track_id)

        track_history[track_id] = Point(cx, cy)


    cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 2)
    cv2.putText(frame, f'IN: {count_in}',   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'OUT: {count_out}', (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()