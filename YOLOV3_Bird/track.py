from collections import deque
from core.test import YoloTest
from core.sort import *
import cv2


yolo = YoloTest()
vs = cv2.VideoCapture("./test/test.mp4")
(W, H) = (None, None)

# ----------------------------------------------------------------------------------------------------------------------

pts = [deque(maxlen=50) for _ in range(999)]
tracker = Sort()
writer = None
memory = {}

# ----------------------------------------------------------------------------------------------------------------------

while True:
    ret, frame = vs.read()

    if not ret:
        break

    bboxes = yolo.predict(frame)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    cv2.putText(frame, "sum --> " + str(len(bboxes)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

        boxes.append([int(coor[0]), int(coor[1]), int(coor[2] - coor[0]), int(coor[3] - coor[1])])
        confidences.append(float(score))
        classIDs.append(class_ind)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

        (x, y) = (int(track[0]), int(track[1]))
        (w, h) = (int(track[2]), int(track[3]))
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)
        center = (int(x + (w - x) / 2), int(y + (h - y) / 2))

        pts[int(track[4])].append(center)
        cv2.circle(frame, center, 1, (0, 0, 255), 5)

        for j in range(1, len(pts[int(track[4])])):
            if pts[int(track[4])][j - 1] is None or pts[int(track[4])][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            cv2.line(frame, (pts[int(track[4])][j - 1]), (pts[int(track[4])][j]), (255, 0, 0), 2)

    cv2.imshow("", frame)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("./out/out.mp4", fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------------------------------------------------------

