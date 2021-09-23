# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
import h5py

flags = tf.app.flags

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'n_frames', '0',
    '.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

flags.DEFINE_string(
    'dataset', 'MSRVTT',
    'Path to the VGGish PCA parameters file.')

FLAGS = flags.FLAGS


import pickle
def main(_):
  info = None
  post = '.mp4'
  if FLAGS.dataset == 'VATEX':
    wav_path = '/work3/yangbang/VATEX/all_wavs'
    # vid2id = pickle.load(open('/home/yangbang/VC_data/VATEX/info_corpus.pkl', 'rb'))['info']['vid2id']
    # info = {v[:11]: k for k,v in vid2id.items()}
    info = None
    base_path = '/work3/yangbang/VATEX/'
  elif FLAGS.dataset == 'MSVD':
    #vid2id = pickle.load(open('/home/yangbang/VC_data/Youtube2Text/info_corpus.pkl', 'rb'))['info']['vid2id']
    #info = vid2id
    wav_path = "/work2/yangbang/MSVD/clip_audio/"
    base_path = "/work2/yangbang/MSVD/"
    post = '.avi'
    info = {}
    lines = open("/work2/yangbang/Youtube2Text/youtube_mapping.txt", 'r').read().strip().split('\n')
    for line in lines:
        _id, vid = line.split()
        info['video%d'%int(vid[3:])] = _id
  else:
    wav_path = '/home/yangbang/VC_data/MSRVTT/all_wavs'
    base_path = '/home/yangbang/VideoCaptioning/MSRVTT/'

  db = h5py.File('./%s_%s.hdf5' % (FLAGS.dataset, FLAGS.n_frames), 'a')

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    for wav_file in os.listdir(wav_path):
        vid = wav_file.split('.')[0]
        if vid in db.keys():
          continue
        examples_batch = vggish_input.wavfile_to_examples2(os.path.join(wav_path, wav_file),  n_frames=int(FLAGS.n_frames), info=info, base_path=base_path, post=post)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        db[vid] = embedding_batch
  db.close()


if __name__ == '__main__':
  tf.app.run()
