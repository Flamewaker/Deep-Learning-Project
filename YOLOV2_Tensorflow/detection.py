import tensorflow as tf
import cv2
from YOLOV2_Tensorflow.model_darknet19 import darknet
from YOLOV2_Tensorflow.decode import decode
from YOLOV2_Tensorflow.utils import preprocess_image, postprocess, draw_detection
from YOLOV2_Tensorflow.config import anchors, class_names
import time
import random
import os


def detect(image_path):
    input_size = (416, 416)
    image_file = image_path

    image = cv2.imread(image_file)
    image_shape = image.shape[:2]       # 只取wh，channel=3不取

    # 加入该语句，第二次检测不会报错。因为第一次计算图已经存在了，再次执行时会和之前已经存在的产生冲突
    tf.reset_default_graph()

    # copy、resize416*416、归一化、在第0维增加存放batchsize维度
    image_cp = preprocess_image(image, input_size)

    # 【1】输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
    tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3])
    model_output = darknet(tf_image) # darknet19网络输出的特征图
    output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍
    output_decoded = decode(model_output=model_output,output_sizes=output_sizes,
                               num_class=len(class_names),anchors=anchors)  # 解码
    # 因为主程序运行路径在UI,用../yolo2_model/yolo2_coco.ckpt
    model_path = "../YOLOV2_Tensorflow/yolo2_model/yolo2_coco.ckpt"
    saver = tf.train.Saver()

    start_time = time.time()

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        bboxes, obj_probs, class_probs = sess.run(output_decoded,feed_dict={tf_image:image_cp})
    # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
    bboxes, scores, class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)

    end_time = time.time()
    detect_time = end_time - start_time

    # 【3】绘制筛选后的边界框
    img_detection, bird_scores = draw_detection(image, bboxes, scores, class_max_index, class_names)

    print(detect_time)
    num = random.randint(1, 1000) + random.randint(1, 100)
    image_name = '../UI/image/' + 'detection-' + str(num) + '.jpg'
    cv2.imwrite(image_name, img_detection)
    print('YOLO_v2 detection has done!')
    return image_name, detect_time, bird_scores
