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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import cv2

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0

def rgb(image):
    return image

def fft(image):
  """Convert to FFT 
  """
  comp_image = tf.cast(image, tf.complex64)
  comp_image = tf.unstack(comp_image,3,2)
  freqim = tf.fft2d(comp_image[0])
  fft_0 = 20*tf.log(tf.complex_abs(freqim)+0.0001)
  freqim = tf.fft2d(comp_image[1])
  fft_1 = 20*tf.log(tf.complex_abs(freqim)+0.0001)
  freqim = tf.fft2d(comp_image[2])
  fft_2 = 20*tf.log(tf.complex_abs(freqim)+0.0001)
  fft_image = tf.stack([fft_0,fft_1,fft_2], axis=2)
  return fft_image

def hsv(image):
  hsv_image = tf.image.rgb_to_hsv(image)
  return hsv_image

def dct(image):
    flt32_image = tf.cast(image, tf.float32)
    flt32_image = tf.unstack(flt32_image,3,2)
    dctim_0 = cv2.dct(flt32_image[0])
    dctim_1 = cv2.dct(flt32_image[1])
    dctim_2 = cv2.dct(flt32_image[2])
    dct_image = tf.stack([dctim_0,dctim_1,dctim_2], axis=2)
    return dct_image


def right_proj(image):
    image32 = tf.cast(image, tf.float32)
    image32 = tf.unstack(image32,3,2)
    s0, u0, v0 = tf.svd(image32[0],full_matrices=True, compute_uv=True)
    s1, u1, v1 = tf.svd(image32[1],full_matrices=True, compute_uv=True)
    s2, u2, v2 = tf.svd(image32[2],full_matrices=True, compute_uv=True)
    pim0 = u0*s0*tf.matrix_inverse(v0)
    pim1 = u1*s1*tf.matrix_inverse(v1)
    pim2 = u2*s2*tf.matrix_inverse(v2)
    proj_image = tf.stack([pim0,pim1,pim2], axis=2)

    return proj_image

def left_proj(image):
    image32 = tf.cast(image, tf.float32)
    image32 = tf.unstack(image32,3,2)
    s0, u0, v0 = tf.svd(image32[0],full_matrices=True, compute_uv=True)
    s1, u1, v1 = tf.svd(image32[1],full_matrices=True, compute_uv=True)
    s2, u2, v2 = tf.svd(image32[2],full_matrices=True, compute_uv=True)
    pim0 = tf.matrix_inverse(u0)*s0*v0
    pim1 = tf.matrix_inverse(u1)*s1*v1
    pim2 = tf.matrix_inverse(u2)*s2*v2
    proj_image = tf.stack([pim0,pim1,pim2], axis=2)

    return proj_image

basis_dict = {0 : rgb,
              1 : fft,
              2 : hsv,
              3 : dct,
              4 : right_proj,
              5 : left_proj,
              }

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def inputs(data_dir, batch_size, basis):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'cifar-10-batches-bin/test_batch.bin')]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  # Image processing for evaluation.

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)
  # Transform the image to desired form.
  mapped_image = basis_dict[basis](resized_image)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(mapped_image)


  # Generate a batch of images and labels by building up a queue of examples.
  return float_image, tf.reshape(read_input.label,[batch_size])

