from UI.DetectPage import DetectPage
from YOLOV2_Tensorflow.detection import detect
from PyQt5.QtGui import QPixmap


# YOLO检测界面
class YOLODtectPage(DetectPage):

    def __init__(self):
        super().__init__()

    def detect_image(self):
        # 检测
        name, time, scores = detect(self.image)
        print(scores)
        print(name)
        self.image_label.setPixmap(QPixmap(name))
        self.time_line.setText(str(time) + '秒')
        self.probability_line.setText(self.process_scores(scores))
        self.bird_num.setText(str(len(scores)))
