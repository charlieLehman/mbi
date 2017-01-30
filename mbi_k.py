# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import confusion_matrix

import mbi

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('rgb_eval_dir', '/home/charlie/mbi_experiment/rgb_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('rgb_checkpoint_dir', '/home/charlie/mbi_experiment/rgb_train', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('rgb_meta', '/home/charlie/mbi_experiment/rgb_train/model.ckpt-0.meta', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")

def main(argv=None):  # pylint: disable=unused-argument

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.rgb_checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver = tf.train.import_meta_graph(FLAGS.rgb_meta)
      saver.restore(sess, ckpt.model_checkpoint_path)
      all_vars = sess.run(conv2) 
      for v in all_vars:
          print(v.name())
    else:
      print('No checkpoint file found')
      return


if __name__ == '__main__':
  tf.app.run()
