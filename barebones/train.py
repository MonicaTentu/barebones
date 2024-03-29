# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import os
from os import listdir
from os.path import isfile, join
import subprocess
import sys
import tensorflow as tf
import boto3
from cfa_utils.s3 import download_dir

# without this PATH append
# it won't find nets - in the /slim directory
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append('slim')

# this is going to execute from the code directory
# so, you don't need code.models, just models
from object_detection.utils import config_util

# process pip installs that won't be part of default environment
def pip_install(package_list):
    for pkg in package_list:
        print ("--> installing:", pkg)
        result = subprocess.call([sys.executable, '-m', 'pip', 'install',pkg])

pip_install(["cython", "pycocotools", "matplotlib"])

## Moving this to an input channel
## Pull down the model
#s3_resource = boto3.resource('s3')
#bucket = "cfa-eadatasciencesb-sagemaker"
#prefix="trained-models/tensorflow_mobilenet/20180718_coco14_mobilenet_v1_ssd300_quantized"
#print(f"Downloading model from: s3://{bucket}/{prefix}")
#def cb(x):
#  print("CB: ", x)
#download_dir(resource=s3_resource, dist=prefix, local='ckpt/', bucket=bucket, status_cb=cb)

from object_detection import model_hparams
from object_detection import model_lib

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
      'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
      'If training data should be evaluated for this job. Note '
      'that one call only use this in eval-only mode, and '
      '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
      'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
      'one of every n train input examples for evaluation, '
      'where n is provided. This is only used if '
      '`eval_training_data` is True.')
flags.DEFINE_string( 'hparams_overrides', None, 'Hyperparameter overrides, '
      'represented as a string containing comma-separated '
      'hparam_name=value pairs.')
flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
      '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
      'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean('run_once', False, 'If running in eval-only mode, whether to run just '
      'one round of eval vs running continuously (default).')
# SageMaker Strings
flags.DEFINE_string('train', os.environ.get('SM_CHANNEL_TRAIN'), 'training data') # None, os.environ.get('SM_CHANNEL_TRAIN'))
flags.DEFINE_string('val', os.environ.get('SM_CHANNEL_VAL'), 'valuation data')  # None, os.environ.get('SM_CHANNEL_VAL'))
flags.DEFINE_string('model_dir', os.environ.get('SM_CHANNEL_MODEL_DIR'), 'output')    # None, os.environ.get('SM_CHANNEL_MODEL_DIR'))

FLAGS = flags.FLAGS

# function to read input path from the config file
# then verify it exists
def check_input_data_existance(pipeline_config_dict):
    input_keys = ['train_input_config', 'eval_input_config']
    for input_key in input_keys:
        print ("checking inputs for:", input_key)
        input_config = pipeline_config_dict[input_key]
        path_list = input_config.tf_record_input_reader.input_path
        for p in path_list:
            exists = tf.io.gfile.exists(p)
            print ("path:", exists, p)

    train_config = pipeline_config_dict['train_config']
    print ("checking for basemodel existance:")
    # Have to fudge this check to get a real filename
    # there are multiple files, this is just a sanity check
    p = train_config.fine_tune_checkpoint + ".meta"
    exists = tf.io.gfile.exists(p)
    print ("path:", exists, p)

def main(unused_argv):
  print ("*** train.py/main()")
  # flags.mark_flag_as_required('model_dir')
  # flags.mark_flag_as_required('pipeline_config_path')

  print ('*** FLAGS ***')
  print ("pipeline_config_path:", FLAGS.pipeline_config_path)
  ## --verification - debug
  print ("config exists:", os.path.exists(FLAGS.pipeline_config_path))
  dir_list = [f for f in listdir(".")]
  for item in dir_list:
        print ("file:", item)

  print ("model_dir:", FLAGS.model_dir)
  print ("train:", FLAGS.train)
  print ("val:", FLAGS.val)
  print ("sample_1_of_n_eval_examples:", FLAGS.sample_1_of_n_eval_examples)
  print ("hparams_overrides:", FLAGS.hparams_overrides)
  print ("checkpoint_dir:", FLAGS.checkpoint_dir)
  # check pipeline config pararameters
  # - input data
  pipeline_config_dict = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path)
  check_input_data_existance(pipeline_config_dict)
  print (" - - - - - - - - -")
  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
  
  tf.enable_eager_execution()
  tf.set_random_seed(0)
  tf.logging.set_verbosity(tf.logging.ERROR)

  # Creates `Estimator`, input functions, and steps
  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples))
  # so here are the outputs (that were in a dict)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  if FLAGS.checkpoint_dir:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         steps=None,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
  tf.app.run()
