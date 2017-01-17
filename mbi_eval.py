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
tf.app.flags.DEFINE_integer('num_examples', 100,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(checkpoint_dir, saver, summary_writer, top_k_op, summary_op, conf_mat, guess, labels, basis):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
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
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      guess_stream = np.array([]) 
      label_stream = np.array([]) 
      confusion = np.zeros(np.shape(conf_mat.eval()))
      while step < num_iter and not coord.should_stop():

        guess_stream = np.append(guess_stream,guess.eval())
        label_stream = np.append(label_stream,labels.eval())

        predictions = top_k_op.eval()

        confusion += conf_mat.eval()
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      print(confusion)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return guess_stream, label_stream


def evaluate(basis,eval_dir, checkpoint_dir):
  """Eval CIFAR-10 for a number of steps."""

  if tf.gfile.Exists(eval_dir):
      tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)

  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'

    images, labels = mbi.inputs(eval_data=eval_data, basis=basis)

    labels =  tf.Print(labels,[labels],message="Label: ")
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = tf.nn.softmax(mbi.inference(images))

    guess = tf.argmax(logits,1)
    guess = tf.Print(guess, [guess], message="Guess: ")

    # Build Confusion Matrix
    conf_mat = tf.contrib.metrics.confusion_matrix(labels, guess)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        mbi.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
      guess_stream, label_stream = eval_once(checkpoint_dir, saver, summary_writer, top_k_op, summary_op, conf_mat,guess, labels, basis)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
    return guess_stream, label_stream




def main(argv=None):  # pylint: disable=unused-argument

  rgb_logits, rgb_labels = evaluate(0, FLAGS.rgb_eval_dir, FLAGS.rgb_checkpoint_dir)
  #fft_logits, fft_labels = evaluate(1, FLAGS.fft_eval_dir, FLAGS.fft_checkpoint_dir)
  #hsv_logits, hsv_labels = evaluate(2, FLAGS.hsv_eval_dir, FLAGS.hsv_checkpoint_dir)
  #dct_logits, dct_labels = evaluate(3, FLAGS.dct_eval_dir)
  #right_proj_logits, right_proj_labels = evaluate(4, FLAGS.right_proj_eval_dir, FLAGS.right_proj_checkpoint_dir)
  #left_proj_logits, left_proj_labels = evaluate(5, FLAGS.left_proj_eval_dir, FLAGS.left_proj_checkpoint_dir)

  #assert np.logical_and(np.equal(rgb_labels, fft_labels)), 'Label Mismatch'

  #late_fuse = np.argmax(0.7*rgb_logits+0.1*fft_logits+0.1*hsv_logits+0.05*right_proj_logits+0.05*left_proj_logits, axis=2)

  #late_fuse = np.argmax(rgb_logits, axis=2)

  #print(rgb_labels[0])
  #print(np.shape(rgb_labels))

  #print(rgb_logits)
  #print(rgb_labels)

  #prediction_performance = np.equal(rgb_logits.astype(int), rgb_labels.astype(int)) + 0

  #print(np.sum(prediction_performance)/np.shape(rgb_labels))





if __name__ == '__main__':
  tf.app.run()
