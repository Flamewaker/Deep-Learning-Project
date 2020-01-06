from UI.DetectPage import DetectPage
from MRAK_RCNN_Tensorflow.samples.demo import detect
from PyQt5.QtGui import QPixmap


# Mask RCNN检测界面
class MRCNNDetectPage(DetectPage):

    def __init__(self):
        super().__init__()

    def detect_image(self):
        name, time, scores = detect(self.image)
        self.image_label.setPixmap(QPixmap(name))
        self.time_line.setText(str(time) + '秒')
        self.probability_line.setText(self.process_scores(scores))
        self.bird_num.setText(str(len(scores)))
