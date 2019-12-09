from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.zz import exporter
import tensorflow as tf

# -------------------------------------------------------------------------------------------------------

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                                                  'one of [`image_tensor`, `encoded_image_string_tensor`, '
                                                  '`tf_example`]')
flags.DEFINE_string('input_shape', None,
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('write_inference_graph', False,
                     'If true, writes inference graph to disk.')
FLAGS = flags.FLAGS


def main(_):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile("./data/ssd_mobilenet_v2_coco.config", 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(
        FLAGS.input_type,
        pipeline_config,
        './training/model/model.ckpt-365999',
        './training/person/',
        input_shape=input_shape,
        write_inference_graph=FLAGS.write_inference_graph)


if __name__ == '__main__':
    tf.app.run()
