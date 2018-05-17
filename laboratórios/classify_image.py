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

"""Classificação de imagens usando o modelo Inception treinado no dataset ImageNet 2012.

Cria um grafo a partir de uma definição de protocolo 'GraphDef' e realiza classificação
de um arquivo JPEG. Produz uma saída com as 5 melhores escolhas juntamente com suas
probabilidades.

Altere o parâmetro --image_file para qualquer imagem armazenada no seu computador
para realizar a classificação daquela imagem.

Observe o tutorial no link abaixo para uma descrição mais detalhada sobre reconhecimento
de imagens:
  https://tensorflow.org/tutorials/image_recognition/

Arquivo adaptado do original

  https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

# pylint: disable=line-too-long
MODEL_DIR = "imagenet/"
NUM_TOP_PREDICTIONS = 5
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converte IDs numéricos em classes."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
        MODEL_DIR, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Carrega os nomes dos alvos em formato de string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Carrega o mapeamento de ID (string) para string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Carrega o mapeamento de ID (string) para integer
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Carrega o mapeamento de integer para string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: {}'.format(val))
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Cria o grafo a partit de um arquivo 'GraphDef' e retorna uma classe 'Saver'."""
  with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Realiza a classificação da imagem.

  Args:
    image: arquivo de imagem

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist {}'.format(image))
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Cria o grafo
  create_graph()

  with tf.Session() as sess:
    # Alguns 'tensors' úteis:
    # 'softmax:0': Uma matriz contendo os scores de classificação entre 1000 labels.
    # 'pool_3:0': Uma matriz representando a penúltima camada de extrator de features.
    # 'DecodeJpeg/contents:0': Uma matriz contendo codificação da imagem
    # Realiza a classificação ao inserir uma imagem no grafo.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Cria um vértice ID --> classe.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
    print("\n")
    for k, node_id in enumerate(top_k):
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('#{:d} - {:s} (score = {:.5f})'.format(k + 1, human_string, score))
    print("\n")

def maybe_download_and_extract():
  """Faz download do model e extrai o arquivo."""
  dest_directory = MODEL_DIR
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      print('\r>> Downloading {:s} {:.1f}'.format(
        filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(MODEL_DIR, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--image_file',
    type=str,
    default='',
    help='Caminho para o arquivo de imagem.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
