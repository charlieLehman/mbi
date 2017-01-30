# Copyright 2015 Google Inc. All Rights Reserved.
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

import logging
import cv2
import numpy as np
import tensorflow as tf
from bisect import bisect_right
from math import ceil
from six.moves import xrange
import os.path as op

import mbi
import mbi_input


FLAGS = tf.app.flags.FLAGS

def _prepare_patch (img, response, y, x, dst_height, scale,
                    stride, accum_padding, half_receptive_field):
  '''Scale patch, overlay receptive field, and response
  '''
  COLOR = (256,256,256)
  THICKNESS = 2
  
  # resize image
  img = cv2.resize(img, dsize=(0,0), fx=scale, fy=scale,
                   interpolation=cv2.INTER_NEAREST)

  # overlay response value
  cv2.putText(img, '%0.1f' % response, 
              org=(0,int(dst_height*0.9)),
              fontFace=cv2.FONT_HERSHEY_DUPLEX, 
              fontScale=dst_height*0.008, 
              color=(0,0,0),
              thickness=THICKNESS+2)

  cv2.putText(img, '%0.1f' % response, 
              org=(0,int(dst_height*0.9)),
              fontFace=cv2.FONT_HERSHEY_DUPLEX, 
              fontScale=dst_height*0.008, 
              color=COLOR,
              thickness=THICKNESS)


  # show the receptive field of a channel (if a user cared to pass params)
  if accum_padding is None or half_receptive_field is None or stride is None:
    logging.warning ('support displaying receptive field only with user input')
  else:
    x_min = y * stride + accum_padding - half_receptive_field
    x_max = y * stride + accum_padding + half_receptive_field
    y_min = x * stride + accum_padding - half_receptive_field + 1
    y_max = x * stride + accum_padding + half_receptive_field + 1
    x_min = int(x_min*scale)
    x_max = int(x_max*scale)
    y_min = int(y_min*scale)
    y_max = int(y_max*scale)
    cv2.rectangle(img, (x_min,y_min), (x_max,y_max), 
                  color=(0,0,0), 
                  thickness=THICKNESS+2)
    cv2.rectangle(img, (x_min,y_min), (x_max,y_max), 
                  color=COLOR, 
                  thickness=THICKNESS)
  return img



def visualize_conv     (sess, images, layer, channels,
                        half_receptive_field=None,
                        accum_padding=None,
                        stride=None,
                        num_excitations=16,
                        num_images=1024,
                        dst_height=96):
  '''
  TL;DR: display some 'images' that receive the strongest response 
    from user-selected 'channels' of a convolutional 'layer'.

  A 64-channel convolutional layer is consists of 64 filters.
  For each of the channels, the corresponding filter naturally fires diffrently
    on different pixels of different images. We're interested in highest responses.
  For each filter, this function searches for such high responses, plus
    the corresponding images and the coordinates of those responses.
  We collect 'num_excitations' images for each filter and stack them into a row.
    Rows from all filters of interest are stacked vetically into the final map.
    For each image, the response value and the receptive field are visualized.

  Args:
    sess:            tensorflow session
    images:          tensor for source images
    layer:           tensor for a convolutional layer response
    channels:        ids of filters of interest, a numpy array.
                       Example: channels=np.asarray([0,1,2]) will result in 3 rows
                       with responses from 0th, 1st, and 2nd filters.
    half_receptive_field:  integer, half of the receptive field for this layer, [1]
    accum_padding:   integer, accumulated padding w.r.t pixels of the input image.
                       Equals 0 when all previous layers use 'SAME' padding
    stride:          integer, equals to multiplication of strides of all prev. layers.
    num_excitations: number of images to collect for each channel
    num_images:      number of input images to search
    dst_height:      will resize each image to have this height
  Returns:
    excitation_map:   a ready-to-show image, similar to R-CNN paper.
  '''
  assert isinstance(channels, np.ndarray), 'channels must be a numpy array'
  assert len(channels.shape) == 1, 'need 1D array [num_filters]'

  # now shape is [im_id, Y, X, ch]
  assert   layer.get_shape()[0].value == FLAGS.batch_size
  Y      = layer.get_shape()[1].value
  X      = layer.get_shape()[2].value
  num_ch = layer.get_shape()[3].value
  logging.info ('Y: %d, X: %d, num_ch: %d' % (Y, X, num_ch))

  # to shape [ch, Y, X, im_id], because we'll reduce on Y, X, and im_id
  layer0 = tf.transpose(layer, (3,1,2,0))
  layer1 = tf.reshape(layer0, [num_ch, -1])
  # indices of the highest responses across batch, X, and Y
  responses, best_ids = tf.nn.top_k(layer1, k=1)

  # make three lists of empty lists
  resps = [list([]) for _ in xrange(len(channels))]
  imges = [list([]) for _ in xrange(len(channels))]
  yx    = [list([]) for _ in xrange(len(channels))]

  # Start the queue runners.
  coord = tf.train.Coordinator()
  try:
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))

    # the same as in cifar10_eval, split evaluation by batches
    num_iter = int(ceil(num_images / FLAGS.batch_size))
    for step in range(num_iter):
      logging.debug ('==========')
      logging.info ('step %d out of %d' % (step, num_iter))

      if coord.should_stop():
        break

      best_ids_vec, images_vec, responses_vec = \
              sess.run([best_ids, images, responses])

      # after this point everything is numpy and opencv

      # collect best responding image from the batch for each filter=channel
      for ch_id, ch in enumerate(channels):
        logging.debug ('----------')
        logging.debug ('ch_id: %d, ch: %s' % (ch_id, ch))

        best_response = responses_vec [ch,0]
        best_id       = best_ids_vec  [ch,0]
        logging.debug ('best_id: %d, Y: %d, X: %d' % (best_id, Y, X))
        # undo reshape -- figure out best indices in Y,X,batch_id coordinates
        best_im = best_id % FLAGS.batch_size
        best_y  = int(best_id / FLAGS.batch_size) / X
        best_x  = int(best_id / FLAGS.batch_size) % X
        # take the image
        best_image = images_vec    [best_im,:,:,:]
        logging.debug ('best_im,best_y,best_x: %d,%d,%d, best_response: %f' % 
                       (best_im, best_y, best_x, best_response))

        # look up the insertion point in the sorted responses lists
        i = bisect_right (resps[ch_id], best_response)
        
        # if the previous response is exactly the same, the image must be same too
        if i > 0 and resps[ch_id][i-1] == best_response:
          logging.debug ('got same response. Skip.')
          continue
        
        # insert both response and image into respective lists
        resps[ch_id].insert(i, best_response)
        imges[ch_id].insert(i, best_image)
        yx[ch_id].insert   (i, (best_y, best_x))

        # pop_front if lists went big and added response is better than current min
        if len(resps[ch_id]) > num_excitations:
          del resps[ch_id][0]
          del imges[ch_id][0]
          del    yx[ch_id][0]

        logging.debug (resps)

  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)


  # scale for resizing images
  src_height = images.get_shape()[1].value
  scale = float(dst_height) / src_height

  for ch_id, _ in enumerate(channels):
    for img_id, img in enumerate(imges[ch_id]):

      imges[ch_id][img_id] = _prepare_patch(
            imges[ch_id][img_id], resps[ch_id][img_id], 
            yx[ch_id][img_id][1], yx[ch_id][img_id][0], 
            dst_height, scale,
            stride, accum_padding, half_receptive_field)

    # concatenate images for this channel
    imges[ch_id]  = np.concatenate(list(imges[ch_id]), axis=1)
  # concatenate stripes of all channels into one map
  excitation_map = np.concatenate(list(imges), axis=0)

  return excitation_map





def visualize_pooling  (sess, images, layer, neurons,
                        half_receptive_field=None,
                        accum_padding=None,
                        stride=None,
	                      num_excitations=16,
                        num_images=1024,
                        dst_height=96):
  '''
  TL;DR: display some 'images' that receive the strongest response 
    from user-selected neurons of a pooling 'layer'.

  A pooling layer is of shape Y x X x Channels.
    Each neuron from that layer is connected to a pixel in the output feature map.
    This function visualizes what a neuron have learned by displying images 
      which receive the strongest responses on that neuron.
    We collect 'num_excitations' images for each neuron and stack them into a row.
      Rows from all neurons of interest are stacked vetically into the final map.
      For each image, the response value and the receptive field are visualized.

  Args:
    sess:            tensorflow session
    images:          tensor for source images
    layer:           tensor for a convolutional layer response
    neurons:         neurons to see best excitations for. 
                      It's probably only a fraction of the layer neurons.
                      Example: neurons=np.asarray([[0,1,2],[58,60,4]])
    half_receptive_field:  integer, half of the receptive field for this layer, [1]
    accum_padding:   integer, accumulated padding w.r.t pixels of the input image.
                       Equals 0 when all previous layers use 'SAME' padding
    stride:          integer, equals to multiplication of strides of all prev. layers.
    num_excitations: number of images to collect for each channel
    num_images:      number of input images to search
    dst_height:      will resize each image to have this height
  Returns:
    excitation_map:   a ready-to-show image, similar to R-CNN paper.

  * Suggestions on how to automatically infer half_receptive_field, accum_padding,
    and stride are welcome.
  '''
  assert isinstance(neurons, np.ndarray), 'neurons must be a numpy array'
  assert len(neurons.shape) == 2 and neurons.shape[1] == 3, 'need shape [N,3]'

  # indices of the "most exciting" patches in a batch, for each neuron
  _, best_ids = tf.nn.top_k(tf.transpose(layer, (1,2,3,0)), k=1)

  # make two lists of empty lists
  # will store num_excitations of best layer/images for each neuron
  resps = [list([]) for _ in xrange(len(neurons))]
  imges = [list([]) for _ in xrange(len(neurons))]

  # Start the queue runners.
  coord = tf.train.Coordinator()
  try:
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))

    # the same as in cifar10_eval, split evaluation by batches
    num_iter = int(ceil(num_images / FLAGS.batch_size))
    for step in range(num_iter):
      logging.debug ('==========')
      logging.info ('step %d out of %d' % (step, num_iter))

      if coord.should_stop():
        break

      best_ids_mat, images_mat, responses_mat = sess.run(
               [best_ids, images, layer]) 

      # after this point everything is numpy and opencv

      # collect best responding image from the batch for each neuron=[y,x,ch]
      for n_id, n in enumerate(neurons):
        logging.debug ('----------')
        logging.debug ('n_id: %d, n: %s' % (n_id, str(n)))

        best_id       = best_ids_mat  [n[0],n[1],n[2],0]
        best_image    = images_mat    [best_id,:,:,:]
        best_response = responses_mat [best_id,n[0],n[1],n[2]]
        logging.debug ('best_id: %d, best_response: %f' % (best_id, best_response))

        # look up the insertion point in the sorted responses lists
        i = bisect_right (resps[n_id], best_response)
        
        # if the previous response is exactly the same, the image must be same too
        if i > 0 and resps[n_id][i-1] == best_response:
          logging.debug ('got same response. Skip.')
          continue
        
        # insert both response and image into respective lists
        resps[n_id].insert(i, best_response)
        imges[n_id].insert(i, best_image)

        # pop_front if lists went big and added response is better than current min
        if len(resps[n_id]) > num_excitations:
          del resps[n_id][0]
          del imges[n_id][0]

        logging.debug (resps)

  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)


  # scale for resizing images
  src_height = images.get_shape()[1].value
  scale = float(dst_height) / src_height

  for n_id, n in enumerate(neurons):
    for img_id, img in enumerate(imges[n_id]):

      imges[n_id][img_id] = _prepare_patch(
            imges[n_id][img_id], resps[n_id][img_id], 
            n[1], n[0], 
            dst_height, scale,
            stride, accum_padding, half_receptive_field)

    # concatenate images for this neuron, and then all the resultant stripes
    imges[n_id]  = np.concatenate(list(imges[n_id]), axis=1)
  excitation_map = np.concatenate(list(imges), axis=0)

  return excitation_map

def put_kernels_on_grid(kernel,grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    return tf.image.convert_image_dtype(x7, dtype = tf.uint8) 

def pool_saturation_map(pool, images, index):
    '''
    Args:
      pool:            tensor for source max pool
      images:          tensor for source images
      index:           int for which image to use in batch
    Returns:
      saturation_map:  composite image of all max poolings overlayed 
                       on original image above the same image where 
                       concentrations of 0 values indicate pooling
                       saturation
    '''
    # Visualize convolution 1 poolng for first image in batch
    pool1_val = tf.reshape(pool[index],[12,12,64])
    pool1_1 = tf.image.resize_images(pool1_val,[24,24])
    pool1_2 = tf.reshape(pool1_1,[24,24,1,64])
    image_vec = tf.tile(tf.reshape(images[index],[24,24,3,1]),[1,1,1,64])
    grid_y = grid_x = 8
    image_grid = put_kernels_on_grid(image_vec, grid_y, grid_x)
    red_grid = tf.to_float(tf.unpack(image_grid, axis=3),name='ToFloat')/255
    image_lg = tf.to_float(tf.image.resize_images(images[0],[208,208],method=0))
    pool1_grid = tf.reshape(tf.to_float(put_kernels_on_grid(pool1_2, grid_y, grid_x),name='ToFloat')/255,[1,208,208])
    packed_grid = tf.pack([tf.add(.6*red_grid[0],10*pool1_grid),.6*red_grid[1],.6*red_grid[2]],axis=3)
    highlight_grid = tf.reshape(tf.fake_quant_with_min_max_args(packed_grid,min=0,max=1),[208,208,3])
    composite_grid = tf.concat(0,[highlight_grid,tf.fake_quant_with_min_max_args(image_lg,min=0,max=1)])

    return composite_grid

def save_vis(train_dir, layer, basis,step, visualization):
    '''
    Args:
      train_dir:         directory to save the visualization 
      layer:             which layer in the model that is
                         visualized
      basis:             which basis is used in the model
      step:              which step in training
      visualization:     image to be saved
    '''
    imdir = op.join(train_dir,layer)
    if not tf.gfile.Exists(imdir):
      tf.gfile.MakeDirs(imdir)
    imdir = op.join(imdir,'%s_%s_%s%s' % (mbi_input.basis_dict[basis].__name__,step,layer,'.png'))
    if np.shape(visualization)[2] > 1:
        visualization = 255*cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    else:
        visualization = 255*visualization

    cv2.imwrite(imdir,visualization)
    print('%s image has been saved in %s' % (layer, imdir))
