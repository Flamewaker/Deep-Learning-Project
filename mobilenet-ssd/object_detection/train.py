from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from object_detection.zz import model_hparams
from object_detection.zz import model_lib
import tensorflow as tf


# -------------------------------------------------------------------------------------------------------

def train():
    config = tf.estimator.RunConfig(model_dir="./training/model/")
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path="./data/ssd_mobilenet_v2_coco.config",
        train_steps=10000000,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=5)

    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    train()

# -------------------------------------------------------------------------------------------------------
