from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import tensorflow as tf
from PIL import Image
import numpy as np
import imutils
import time
import cv2

# -------------------------------------------------------------------------------------------------------

PATH_TO_FROZEN_GRAPH = './training/person/frozen_inference_graph.pb'
PATH_TO_LABELS = "./data/person.pbtxt"

# -------------------------------------------------------------------------------------------------------

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# -------------------------------------------------------------------------------------------------------

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# -------------------------------------------------------------------------------------------------------

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        cap = cv2.VideoCapture('./test_data/test.mp4')
        # cap = cv2.VideoCapture(0)
        _, image = cap.read()
        start = time.time()
        frame_count = 0

        # -------------------------------------------------------------------------------------------------------

        ops = detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

        # -------------------------------------------------------------------------------------------------------

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=450)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            (im_width, im_height) = image.size
            image_np = np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)
            image = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            # ------------------------------------------------------------------

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=3)

            # ------------------------------------------------------------------

            end = time.time()
            frame_count += 1
            FPS = frame_count / (end - start)
            image_np = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)
            cv2.putText(image_np, "FPS:" + "%0.2f" % FPS, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("", image_np)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

# -------------------------------------------------------------------------------------------------------
