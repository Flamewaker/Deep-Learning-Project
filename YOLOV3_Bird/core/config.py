from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "../YOLOV3_Bird/data/classes/bird.names"
__C.YOLO.ANCHORS                = "../YOLOV3_Bird/data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "../YOLOV3_Bird/checkpoint/yolov3_bird.ckpt"
__C.YOLO.DEMO_WEIGHT            = "../YOLOV3_Bird/checkpoint/yolov3_bird.ckpt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "../YOLOV3_Bird/data/dataset/train.txt"
__C.TRAIN.BATCH_SIZE            = 6
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-4
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 1000
__C.TRAIN.SECOND_STAGE_EPOCHS   = 1000
__C.TRAIN.INITIAL_WEIGHT        = "../YOLOV3_Bird/checkpoint/yolov3_bird.ckpt"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "../YOLOV3_Bird/data/dataset/test.txt"
__C.TEST.BATCH_SIZE             = 6
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "../YOLOV3_Bird/data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "../YOLOV3_Bird/checkpoint/yolov3_bird.ckpt"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.45
__C.TEST.IOU_THRESHOLD          = 0.45





