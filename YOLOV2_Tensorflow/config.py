import os

# file path
DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')   #存放输出文件的地方，data/pascal_voc/output

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')  #weights_dir, 路径为data/pascal_voc/weights

WEIGHTS_FILE = None   #weights file
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']    #PASCAL VOC数据集的20个类别

FLIPPED = True


def read_coco_labels():
    f = open("../YOLOV2_Tensorflow/yolo2_data/coco_classes.txt")
    class_names = []
    for l in f.readlines():
        l = l.strip() # 去掉回车'\n'
        class_names.append(l)
    return class_names

class_names = read_coco_labels()

anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

# model parameter
IMAGE_SIZE = 416

CELL_SIZE = 13

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

# solver parameter

GPU = '2'

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45

MAX_ITER = 15000

SUMMARY_ITER = 10

SAVE_ITER = 1000

# test parameter
THRESHOLD = 0.3

IOU_THRESHOLD = 0.5