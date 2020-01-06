from UI.DetectPage import DetectPage
from Faster_RCNN_Tensorflow.demo import bird_detection
from PyQt5.QtGui import QPixmap
import os
import sys


# faster rcnn检测界面
class FRCNNDetectPage(DetectPage):

    def __init__(self):
        super().__init__()

    def detect_image(self):
        print(self.image)
        if self.image == './image/bird.jpg':
            self.image = os.path.join(sys.path[0], 'image/bird.jpg')
        name, time, scores = bird_detection(self.image)
        self.image_label.setPixmap(QPixmap(name))
        self.time_line.setText(str(time) + '秒')
        self.probability_line.setText(self.process_scores(scores))
        self.bird_num.setText(str(len(scores)))

