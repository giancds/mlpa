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

"""Rede neural convolucional similar ao LeNet-5 para reconhecimento de imagens usando o dataset
  MNIST.

  Baseado nos seguintes tutoriais:

  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
  https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# hyperparâmetros do modelo
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # tamanho do dataset de validação
SEED = 66478  # semente para o gerador de números aleatórios
BATCH_SIZE = 64  # tamanho do batch de treino
NUM_EPOCHS = 1  # número de épocas
EVAL_BATCH_SIZE = 64  # tamanho do batch de validação/teste
EVAL_FREQUENCY = 100  # Número de batches entre cada validação/teste

FLAGS = None


def maybe_download(filename):
  """Faz download dos dados caso não estejam no diretório correto. """
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extrai as imagens em um tensor 4D [image index, y, x, channels].

  Pixels são normalizados do intervalo [0, 255] para o intervalo [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extrai os alvos em um vetor de IDs no formato int64."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def error_rate(predictions, labels):
  """Retorna a taxa de erros."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def create_model(data_format):
  """Constrói a rede neural convolucional usando a API tf.keras para uso em grafos.

  Args:
    data_format: 'channels_first' ou 'channels_last'. 'channels_first' é mais rápido
      em GPUs enquanto 'channels_last' deve ser utilizado com CPUs. Para mais detalhes:
      https://www.tensorflow.org/performance/performance_guide#data_formats
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


def main(_):
  # obtém os datasets
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
  test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

  # Extrai em arrays usando a biblioteca numpy
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)
  test_data = extract_data(test_data_filename, 10000)
  test_labels = extract_labels(test_labels_filename, 10000)

  # cria um dataset de validação
  validation_data = train_data[:VALIDATION_SIZE, ...]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:, ...]
  train_labels = train_labels[VALIDATION_SIZE:]
  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # Aqui é onde os inputs e os alvos são inseridos no grafo.
  # Os vértices do tipo placeholders recebem os exemplos do batch a cada etapa
  # usando o argumento {feed_dict} na chamada ao método Run() abaixo
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  model = create_model(data_format="channels_last")

  # cálculo da saída da rede neural e da função de custo 'cross-entropy'
  logits = model(train_data_node, training=True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # Otimizador: cria uma variável que é incrementada a cada batch e controla
  # a redução da taxa de aprendizado (learning_rate)
  batch = tf.Variable(0, dtype=tf.float32)
  # Reduz uma vez por época, usando uma redução exponencial que começa em 0.01
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Utiliza o otimizado 'momentum'
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Faz a classificação para o batch usado para o treino
  train_prediction = tf.nn.softmax(logits)

  # Faz a classificação (menos frequente) para teste e validação
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Função utilitária para realizar a classificação também em batches,
  # passados para {eval_data} e obtém os resultados de {eval_predictions}.
  # Reduz consumo de memória e habilita o uso de GPUs com menos memória.
  def eval_in_batches(data, sess):
    """Obtém a classificação para um dataset usando batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: {:d}" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Cria uma sessão local para o treinamento da rede neural
  start_time = time.time()
  with tf.Session() as sess:
    # Inicializa os parâmetros do modelo.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Laço que itera pelos exemplos de treino.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Calcula um offset para o batch em processamento.
      # Randomizar os exemplos de treino seria uma alternativa
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # Esta variável do tipo 'dictionary' mapeia os inputs (recebidos como arrays do numpy) para
      # os vértices do grafo que devem recebê-los.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Realiza os passos de backprop e atualiza os parâmetros do modelo
      sess.run(optimizer, feed_dict=feed_dict)
      # imprime algumas informações quando atingirmos o ponto de validação
      if step % EVAL_FREQUENCY == 0:
        # busca dados 'extras' de alguns dos vértices
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step {:d} (epoch {:.2f}), {:.1f} ms'.format(
          step, float(step) * BATCH_SIZE / train_size,
          1000 * elapsed_time / EVAL_FREQUENCY))
        print('Custo do Minibatch: {:.3f}, learning rate: {:.6f}'.format(l, lr))
        print('Erro no Minibatch: {:.1f}'.format(error_rate(predictions, batch_labels)))
        print('Erro de validação: {:.1f}'.format(error_rate(
            eval_in_batches(validation_data, sess), validation_labels)))
        sys.stdout.flush()
    # Ao final, imprime os resultados
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Erro no teste: {:.1f}'.format(test_error))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
