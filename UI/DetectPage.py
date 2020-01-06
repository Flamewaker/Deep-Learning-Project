import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton, QLabel, QHBoxLayout, QBoxLayout, QFormLayout, QLineEdit, QFileDialog
from PyQt5.QtGui import *


# 检测界面父类
class DetectPage(QWidget):

    # 界面绘制
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('./config/HomePageIcon.png'))  # 设置窗口的图标
        self.set_center()
        self.image = './image/bird.jpg'

        # 第一行
        self.upload_button = QPushButton('上传图片')
        self.detect_button = QPushButton('开始检测')

        h_layout = QHBoxLayout()
        h_layout.setSpacing(15)
        h_layout.addWidget(self.upload_button)
        h_layout.addWidget(self.detect_button)

        self.upload_button.clicked.connect(self.pick_image)       # 绑定事件
        self.detect_button.clicked.connect(self.detect_image)     # 绑定事件

        # 第二行
        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap(self.image))

        # 第三行
        self.time_line = QLineEdit()
        self.time_line.setEnabled(False)
        self.probability_line = QLineEdit()
        self.probability_line.setEnabled(False)
        self.bird_num = QLineEdit()
        self.bird_num.setEnabled(False)

        form_layout = QFormLayout()
        form_layout.addRow('检测时间：', self.time_line)
        form_layout.addRow('score值：', self.probability_line)
        form_layout.addRow('鸟个数：', self.bird_num)

        # 整体布局
        self.main_layout = QBoxLayout(QBoxLayout.Down)
        self.main_layout.addLayout(h_layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(form_layout)
        self.setLayout(self.main_layout)

    # 从本地选择图片
    def pick_image(self):
        image_name, image_type = QFileDialog.getOpenFileName(self, "选择图片", "", "*.jpg;;*.png;;")
        self.image = image_name
        self.image_label.setPixmap(QPixmap(self.image))

    # 检测图片,供子类继承
    def detect_image(self):
        pass

    def process_scores(self, scores):
        result = '['
        if len(scores) != 0:
            for score in scores:
                result = result + str('%.3f' % score) + ','
            result = result + ']'
        else:
            result = '没有检测到任何鸟群'
        return result


    # 控制主页显示在屏幕中心
    def set_center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序和对象
    home_page = DetectPage()
    home_page.show()  # 显示窗口
    sys.exit(app.exec_())