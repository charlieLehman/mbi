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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
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
tf.app.flags.DEFINE_string('fft_eval_dir', '/home/charlie/mbi_experiment/fft_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('hsv_eval_dir', '/home/charlie/mbi_experiment/hsv_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('dct_eval_dir', '/home/charlie/mbi_experiment/dct_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('right_proj_eval_dir', '/home/charlie/mbi_experiment/right_proj_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('left_proj_eval_dir', '/home/charlie/mbi_experiment/left_proj_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('rgb_checkpoint_dir', '/home/charlie/mbi_experiment/rgb_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('fft_checkpoint_dir', '/home/charlie/mbi_experiment/fft_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('hsv_checkpoint_dir', '/home/charlie/mbi_experiment/hsv_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('dct_checkpoint_dir', '/home/charlie/mbi_experiment/dct_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('right_proj_checkpoint_dir', '/home/charlie/mbi_experiment/right_proj_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('left_proj_checkpoint_dir', '/home/charlie/mbi_experiment/left_proj_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")

def eval_once(checkpoint_dir, saver, summary_writer, summary_op, basis, labels, logits, conf_mat):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      guess_stream = np.zeros((num_iter,FLAGS.batch_size,mbi.NUM_CLASSES))
      label_stream = np.zeros((num_iter,FLAGS.batch_size)) 
      confusion = np.zeros([mbi.NUM_CLASSES,mbi.NUM_CLASSES])
      while step < num_iter and not coord.should_stop():
        logitss, labelss, confusions = sess.run([logits, labels, conf_mat])
        label_stream[step] = labelss
        guess_stream[step] = logitss
        confusion += confusions
        step += 1

      print(confusion)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return guess_stream, label_stream, confusion


def evaluate(basis,eval_dir, checkpoint_dir):
  """Eval CIFAR-10 for a number of steps."""

  if tf.gfile.Exists(eval_dir):
      tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    #eval_data = FLAGS.eval_data == 'test'
    images, labels = mbi.inputs(eval_data=FLAGS.eval_data,basis=basis)


    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = tf.nn.softmax(mbi.inference(images))

    #labels =  tf.Print(labels,[labels],message="Label: ")
    #logits =  tf.Print(logits,[logits],message="Logits: ")

    # Build Confusion Matrix
    conf_mat = tf.contrib.metrics.confusion_matrix(labels, tf.argmax(logits,1))


    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        mbi.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    return eval_once(checkpoint_dir, saver, summary_writer, summary_op,  basis, labels, logits, conf_mat)



def main(argv=None):  # pylint: disable=unused-argument


  rgb_logits, rgb_labels, rgb_conf = evaluate(0, FLAGS.rgb_eval_dir, FLAGS.rgb_checkpoint_dir)
  fft_logits, fft_labels, fft_conf= evaluate(1, FLAGS.fft_eval_dir, FLAGS.fft_checkpoint_dir)
  hsv_logits, hsv_labels, hsv_conf= evaluate(2, FLAGS.hsv_eval_dir, FLAGS.hsv_checkpoint_dir)
  #dct_logits, dct_labels, dct_conf = evaluate(3, FLAGS.dct_eval_dir, FLAGS.dct_checkpoint_dir)
  right_proj_logits, right_proj_labels, rpr_conf = evaluate(4, FLAGS.right_proj_eval_dir, FLAGS.right_proj_checkpoint_dir)
  #left_proj_logits, left_proj_labels, lpr_conf = evaluate(5, FLAGS.left_proj_eval_dir, FLAGS.left_proj_checkpoint_dir)



  assert np.equal(rgb_labels, fft_labels).all(), 'Label Mismatch'

  print("====================")


  key = ['rgb', 'fft', 'hsv', 'dct', 'rpr', 'lpr']
  logits = [rgb_logits, fft_logits,hsv_logits, dct_logits, right_proj_logits, left_proj_logits]

  weights =  [[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]]

  print(key)
  print(fuse(weights, rgb_labels, logits))

  weights =  [[1,1,1,1,1,1],
              [0,1,1,1,1,1],
              [0,0,1,1,1,1],
              [0,0,0,1,1,1],
              [0,0,0,0,1,1]]
  print(fuse(weights, rgb_labels, logits))

  
  

def fuse(weights, eval_labels,logits):
    performance = np.zeros(np.shape(weights)[0]) 
    for k,n in enumerate(weights):
        logit_weights = np.array(n)
        masked_logits = np.zeros(np.shape(logits))
        for i,m in enumerate(logits):
            masked_logits[i] = logit_weights[i]*m
        guess_stream = np.sum(masked_logits, axis=0)
        guess = np.argmax(guess_stream, axis=2)
        guess = np.reshape(guess,np.shape(eval_labels))
        performance_stream = np.equal(guess, eval_labels) + 0
        performance[k] = np.sum(performance_stream/FLAGS.num_examples)
    return performance

if __name__ == '__main__':
  tf.app.run()
