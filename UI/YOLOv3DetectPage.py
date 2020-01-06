from UI.DetectPage import DetectPage
from YOLOV3_Bird.demo_image import detect
from PyQt5.QtGui import QPixmap


# SSD检测界面
class YOLOv3DetectPage(DetectPage):

    def __init__(self):
        super().__init__()

    def detect_image(self):
        # 检测
        name, time, scores = detect(self.image)
        self.image_label.setPixmap(QPixmap(name))
        self.time_line.setText(str(time) + '秒')
        self.probability_line.setText(self.process_scores(scores))
        self.bird_num.setText(str(len(scores)))
