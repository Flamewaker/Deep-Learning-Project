import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QMessageBox, QPushButton, QGridLayout
from PyQt5.QtGui import *
from UI.DetectPage import DetectPage
from UI.YOLODetectPage import YOLODtectPage
from UI.YOLOv3DetectPage import YOLOv3DetectPage
from UI.MRCNNDetectPage import MRCNNDetectPage
from UI.FRCNNDetectPage import FRCNNDetectPage


# 鸟群检测首页
class HomePage(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    # 界面绘制
    def init_ui(self):
        self.set_window_figure()
        layout = self.layout()
        self.setLayout(layout)

    # 设置主页属性
    def set_window_figure(self):
        # 设置窗口的位置和大小
        # self.setGeometry(300, 300, 1000, 750)
        self.resize(500, 100)
        self.setWindowTitle('鸟群检测')  # 设置窗口的标题
        self.setWindowIcon(QIcon('./config/HomePageIcon.png'))  # 设置窗口的图标
        self.set_center()

    def layout(self):
        grid = QGridLayout()
        grid.setSpacing(10)

        # 鸟群检测按钮
        yolo_button = QPushButton('YOLO v2')
        yolov3_button = QPushButton('YOLO v3')
        mrcnn_button = QPushButton('Mask RCNN')
        frcnn_button = QPushButton('Faster RCNN')

        self.yolo_detect = YOLODtectPage()
        self.yolo_detect.setWindowTitle('YOLOv2')
        yolo_button.clicked.connect(self.yolo_detect.show)

        self.yolov3_detect = YOLOv3DetectPage()
        self.yolov3_detect.setWindowTitle('YOLOv3')
        yolov3_button.clicked.connect(self.yolov3_detect.show)

        self.mrcnn_detect = MRCNNDetectPage()
        self.mrcnn_detect.setWindowTitle('Mask RCNN')
        mrcnn_button.clicked.connect(self.mrcnn_detect.show)

        self.frcnn_detect = FRCNNDetectPage()
        self.frcnn_detect.setWindowTitle('Faster RCNN')
        frcnn_button.clicked.connect(self.frcnn_detect.show)
         
        grid.addWidget(yolo_button, 1, 0)
        grid.addWidget(yolov3_button, 1, 1)
        grid.addWidget(mrcnn_button, 2, 0)
        grid.addWidget(frcnn_button, 2, 1)

        return grid

    # 控制主页显示在屏幕中心
    def set_center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 消息框
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出提醒',
                                     "是否退出?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)        # 创建应用程序和对象
    home_page = HomePage()
    home_page.show()                    # 显示窗口
    sys.exit(app.exec_())