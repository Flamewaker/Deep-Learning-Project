from YOLOV3_Bird.core.test import YoloTest
import numpy as np
import tensorflow as tf
import cv2
import time
import random
# ----------------------------------------------------------------------------------------------------------------------

def detect(image_path):
    tf.reset_default_graph()
    yolo = YoloTest()
    image = cv2.imread(image_path)
    bboxes1, detect_time1= yolo.predict(image, 0)
    bboxes, detect_time = yolo.predict(image, 1)

    image_h, image_w, _ = image.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    bird_scores = []
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_mess = '%.3f' % (score)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, (0, 255, 255), 2)
        bird_scores.append(score)
        bbox_mess = '%.3f' % (score)
        cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

    # cv2.imshow("", image)
    num = random.randint(1, 1000) + random.randint(1, 100)
    image_name = '../UI/image/' + 'detection-' + str(num) + '.jpg'
    # image_name = './' + 'detection-' + str(num) + '.jpg'
    cv2.imwrite(image_name, image)
    del yolo
    return image_name, detect_time, bird_scores
# ----------------------------------------------------------------------------------------------------------------------

