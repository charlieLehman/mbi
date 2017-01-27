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
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import mbi
import mbi_input

import cv2
import os.path as op

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('rgb_dir', '/home/charlie/mbi_experiment/rgb_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('fft_dir', '/home/charlie/mbi_experiment/fft_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('hsv_dir', '/home/charlie/mbi_experiment/hsv_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dct_dir', '/home/charlie/mbi_experiment/dct_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dct2_dir', '/home/charlie/mbi_experiment/dct2_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('right_proj_dir', '/home/charlie/mbi_experiment/right_proj_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('left_proj_dir', '/home/charlie/mbi_experiment/left_proj_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train(train_dir, basis):
  """Train CIFAR-10 for a number of steps."""
  print('Begin training %s for %d steps' % (mbi_input.basis_dict[basis].__name__, FLAGS.max_steps))


  with tf.Graph().as_default():

    if tf.gfile.Exists(train_dir):
      ckpt = tf.train.get_checkpoint_state(train_dir)
      global_step = tf.Variable(int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]), trainable=False)
    else:
      global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = mbi.distorted_inputs(basis)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = mbi.inference(images)

    # Visualize conv1 features
    with tf.variable_scope('conv1') as scope_conv:
      tf.get_variable_scope().reuse_variables()
      weights = tf.get_variable('weights')
      grid_x = grid_y = 8   # to get a square grid for 64 conv1 features
      grid = mbi.put_kernels_on_grid (weights, grid_y, grid_x)
      tf.summary.image('conv1/features', grid, max_outputs=1)

    # Calculate loss.
    loss = mbi.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = mbi.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)


    if tf.gfile.Exists(train_dir):
      tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
      start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      tf.gfile.MakeDirs(train_dir)
      start_step = 0

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in xrange(start_step, start_step+FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        img = sess.run(grid2)[0]
        # Save conv1 visualisation
        imdir = op.join(train_dir,'conv1')
        if not tf.gfile.Exists(imdir):
          tf.gfile.MakeDirs(imdir)
        imdir = op.join(imdir,'%s_%s_%s' % (mbi_input.basis_dict[basis].__name__,step,'conv1.png'))
        cv2.imwrite(imdir,img)
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  mbi.maybe_download_and_extract()
  train(FLAGS.rgb_dir,0)
  #train(FLAGS.hsv_dir,2)
  #train(FLAGS.dct_dir,3)
  #train(FLAGS.fft_dir,1)
  #train(FLAGS.right_proj_dir,4)
  #train(FLAGS.left_proj_dir,5)


if __name__ == '__main__':
  tf.app.run()
