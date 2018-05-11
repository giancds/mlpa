from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf

import tensorflow.contrib.eager as tfe
import input_data

def load_data(data_dir):
  """Returns training and test tf.data.Dataset objects."""
  data = input_data.read_data_sets(data_dir, one_hot=True)
  train_ds = tf.data.Dataset.from_tensor_slices((data.train.images,
                                                 data.train.labels))
  test_ds = tf.data.Dataset.from_tensors((data.test.images, data.test.labels))
  return (train_ds, test_ds)

def loss(predictions, labels):
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
#       tf.nn.softmax_cross_entropy_with_logits(
          logits=predictions, labels=labels))

def compute_accuracy(predictions, labels):
  return tf.reduce_sum(
      tf.cast(
          tf.equal(
              tf.argmax(predictions, axis=1,
                        output_type=tf.int64),
              tf.argmax(labels, axis=1,
                        output_type=tf.int64)),
          dtype=tf.float32)) / float(predictions.shape[0].value)

def train_one_epoch(model, optimizer, dataset, log_interval=None):
  """Treina o modelo usando um 'dataset' com  um 'otimizador'."""

  #  esta linha serve para ativar ou buscar um contador interno da biblioteca que retorna o número do
  # 'batch' executado - se existe, busca o existente, senão cria um novo
  tf.train.get_or_create_global_step()

  # este é o laço principal da otimização
  for (batch, (inputs, alvos)) in enumerate(tfe.Iterator(dataset)):
    # esta linha diz para a biblioteca guardar o estado da otimização a cada 10 'batches' contados por
    # tf.train.get_or_create_global_step() criado acima
    # desta forma, podemos visualizar o comportamento da otimização e entendermos como o modelo está funcionando e
    # quais partes podem ser melhoradas
    with tf.contrib.summary.record_summaries_every_n_global_steps(10):
      # nesta linha, definimos uma 'fita' que vai armazenar a ordem de execução da rede neural e assim como
      # os gradientes calculados para backpropagation
      with tfe.GradientTape() as tape:
        prediction = model(inputs, training=True)  # obtendo a saída da rede neural
        loss_value = loss(prediction, alvos)  # calculando o sinal de erro
        tf.contrib.summary.scalar('loss', loss_value)  # aqui, adicionamos o sinal de erro à vizualição
        tf.contrib.summary.scalar('accuracy',
                                  compute_accuracy(prediction, alvos)) # adicionamos também o percentual de acertos
      # é aqui que a 'mágica' acontece
      # tensorflow possui um módulo de auto-diferenciação que lê a 'fita' de execução e calcula todos as derivadas
      # necessárias para o algoritmo de backpropagation, incluindo a execução do próprio algoritmo!
      # desta forma, não precisamos nos preocupar com a lógica daquele algoritmo e podemos nos concentrar
      # no modelo em si
      grads = tape.gradient(loss_value, model.variables)
      # uma vez executado o backpropagation, o optimizer irá aplicar a atualização de parâmetros
      optimizer.apply_gradients(zip(grads, model.variables))
      # por fim, vamos imprimir na tela o progresso da otimização
      if log_interval and batch % log_interval == 0:
        print('Batch #%d\tLoss: %.6f' % (batch, loss_value))

def test(model, dataset):
  """Realiza um teste do 'model' utilizando datapoins retirados do 'dataset'."""
  avg_loss = tfe.metrics.Mean('loss')  # aqui usamos uma função pré-construída para calcular a média do sinal de erro
  accuracy = tfe.metrics.Accuracy('accuracy')  # e fazemos o mesmo com o percentual de acertos

  # este é o laço principal do teste
  for (images, labels) in tfe.Iterator(dataset):
    predictions = model(images, training=False)  # obtemos as predições
    avg_loss(loss(predictions, labels))  # calculamos a média do sinal de erro
    accuracy(tf.argmax(predictions, axis=1, output_type=tf.int64),
             tf.argmax(labels, axis=1, output_type=tf.int64))  # calculamos o percentual de acertos
  print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))  # imprimimos na tela algumas dessas informações

  # por fim, nas linhas abaixo, adicionamos mais dados à visualização do treino
  with tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.scalar('loss', avg_loss.result())
    tf.contrib.summary.scalar('accuracy', accuracy.result())

class MNISTModel(tfe.Network):
    """MNIST Network.

        Para podermos executar nosso modelo no modo eager execution, nossa classe deve obrigatoriamente
            herdar da classe tfe.Network.

    """

    def __init__(self, data_format):
        super(MNISTModel, self).__init__(name='')
        # esta parte define o formato dos dados de input - recomenda-se utilizar o padrão definido no código
        # de treino - para descrições mais detalhadas, veja a documentação da classe tf.layers.Conv2D
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, 28, 28, 1]
        # neste ponto definimos a estrutura da rede neural
        # para que a 'fita" de gradientes mantenha um registro das dependências e a biblioteca mantenha
        # um registro dos parâmetros, devemos passar a definição de cada camada para a função self.track_layer
        # que é definida em tfe.Network
        # perceba que cada camada é definida por uma classe que pertence ao módulo tf.layers
        # para as camadas Conv2d (Convoluções), nós precisamos definir o número de filtros e o tamanho do filtro -
        # para definir o tamanho do filtro podemos passar um número inteiro (define mesma altura /largura) ou uma
        # tupla que definirá altura/largura customizados
        self.conv1 = self.track_layer(
            tf.layers.Conv2D(filters=32, kernel_size=5, data_format=data_format, activation=tf.nn.relu))
        self.conv2 = self.track_layer(
            tf.layers.Conv2D(filters=64, kernel_size=5, data_format=data_format, activation=tf.nn.relu))
        # para definir as camadas tradicionais, selecionamos a camada Dense e definimos o número de neurônios (units)
        # na camada, assim como a função de ativação (activation) - no caso de não passarmos uma função de ativação
        # como parâmetro, a camada se torna uma camada 'linear'- em outras palavras, a camada realiza apenas o cálculo
        # da transformação linear sobre os inputs
        self.fc1 = self.track_layer(tf.layers.Dense(units=1024, activation=tf.nn.relu))
        self.fc2 = self.track_layer(tf.layers.Dense(units=10))
        self.dropout = self.track_layer(tf.layers.Dropout(0.5))
        # a camada de maxpool não possui parâmetros pois ela apelas realiza uma operação 'max' (seleciona o maior
        # valor dentre uma lista de valores) e, desta forma, podemos criar apenas uma camada em nosso código e
        # reutilizá-la para cada Conv2d. para definir esta camada, devemos definir o tamanho da 'janela' de valores e
        # o deslocamento da 'janela' sobre o resultado da convolução. a definição da altura/largura segue o mesmo
        # padrão utilizado na camada Conv2d
        self.max_pool2d = self.track_layer(
            tf.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), padding='SAME', data_format=data_format))

    def call(self, inputs, training):
        """ Classifica os dígitos baseado nos 'inputs'.
        """
        # para termos certeza de que os inputs estão no formato esperado, realizamos uma operação reshape
        x = tf.reshape(inputs, self._input_shape)
        # em seguida, aplicamos em ordem as camadas definidas na inicialização da classe
        x = self.conv1(x)  # primeira convolução
        x = self.max_pool2d(x)  # primeiro max-pool
        x = self.conv2(x)  # segunda convolução
        x = self.max_pool2d(x)  # segundo max-pool - nota: esta é a mesma função acima - ver comentários na definição
        # como nós estamos operando sobre imagens, cada input será 2d (uma matriz). no entanto, as camadas
        # Dense (camadas 'tradicionais') esperam um input 1d para cada exemplo. por isso, nós usamos uma função
        # pré-contruida para fazer um reshape e transformar o input 2d em 1d - nota: nós poderíamos ter utilizado
        # a função reshape como feito anteriormente, passando o formato correto.
        x = tf.layers.flatten(x)
        x = self.fc1(x)  # primeira camada dense
        # aqui aplicamos uma função de dropout para auxiliar na regularização - o parâmetro training serve para
        # ativar/desativar o dropout - ele é aplicado apenas durante o treino e não durante o teste
        x = self.dropout(x, training=training)
        x = self.fc2(x)  #  segunda camada dense (output)
        return x

def main(_):

    # primeiro devemos habilitar o modo eager execution
    # aviso: este comando deve ser executado apenas uma vez. caso contrário um exceção será lançada
    tfe.enable_eager_execution()

    # aqui define-se o dispositivo que será utilizado para o treino da rede e o formato dos dados passados
    # de maneira geral, usamos o formato de dados default como definidos abaixo
    (device, data_format) = ('/gpu:0', 'channels_first')
    if FLAGS.no_gpu or tfe.num_gpus() <= 0:
        (device, data_format) = ('/cpu:0', 'channels_last')
        print('Using device %s, and data format %s.' % (device, data_format))

    # carregando os datasets em treino e teste e embaralhando os exemplos
    # embaralhar os exemplos de treino auxilia a "quebrar" as dependências criadas pelo processamento
    # ordenado do dataset
    (train_ds, test_ds) = load_data(FLAGS.data_dir)
    train_ds = train_ds.shuffle(60000).batch(FLAGS.batch_size)

    # aqui nós criamos a classe contendo o modelo - veja detalhes na seção anterior
    model = MNISTModel(data_format)
    # aqui define-se o algoritmo que fará a otmização do modelo
    optimizer = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)

    # criamos um diretório para armazenar o modelo treinado
    if FLAGS.output_dir:
        train_dir = os.path.join(FLAGS.output_dir, 'train')
        test_dir = os.path.join(FLAGS.output_dir, 'eval')
        tf.gfile.MakeDirs(FLAGS.output_dir)
    else:
        train_dir = None
        test_dir = None
    # aqui definimos o local no qual o programa armazenará os resumos gerados sobre o treino e sobre o teste
    # assim como instância a classe que vai realizar a gravação dos resumos
    summary_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=10000)
    test_summary_writer = tf.contrib.summary.create_file_writer(test_dir, flush_millis=10000, name='test')
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
    epoch = 1
    # esta é a parte principal do programa
    # primeiro selecionamos o dispositivo que será usado para o treino - CPU ou GPU
    with tf.device(device):
        # criamos um laço para controlar as épocas

        # a biblioteca nos fornece diversas funções auxiliares pré-construídas para facilitar
        # a criação dos modelos. no caso abaixo, utilizamos uma função exclusiva do modo eager execution
        # que verifica se temos um modelo já salvo e utiliza para inicializar os parâmetros - pesos - da rede
        # isso é bastante útil quando o programa para no meio do treino e precisamos reiniciar a partir de um
        # certo ponto. no entanto, se temos um modelo salvo e queremos reiniciar do início, precisamos
        # excluir o modelo antigo
        with tfe.restore_variables_on_create(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):
            #  esta linha serve para ativar ou buscar um contador interno da biblioteca que retorna o número do
            # 'batch' executado - se existe, busca o existente, caso contrário cria um novo
            global_step = tf.train.get_or_create_global_step()
            start = time.time()
            # utilizando a classe que grava os resumos do treino
            with summary_writer.as_default():
                # executamos uma época de treino - ver código das funções auxiliares
                train_one_epoch(model, optimizer, train_ds, FLAGS.log_interval)
            end = time.time()
            # feito isso, imprimimos algumas informações na tela para facilitar o acompanhamento
            print('\nTrain time for epoch #%d (global step %d): %f' % (
                epoch, global_step.numpy(), end - start))
            # utilizando a classe que grava os resumos do teste
            with test_summary_writer.as_default():
                # executamos uma época de verificação sobre o dataset de teste
                test(model, test_ds)
                # criamos uma lista com todos os parâmetros do modelo (e do laço de treino) que estamos treinando
                all_variables = (model.variables + optimizer.variables() + [global_step])
                # utilizando mais uma função utilitária (exclusiva do modo eager execution),
                # salvamos em disco os parâmetros do modelo que estamos treinando
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    type=str,
    default='/tmp/tensorflow/mnist/input_data',
    help='Directory for storing input data')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    metavar='N',
    help='Directory to write TensorBoard summaries')
parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default='data/checkpoints/',
    metavar='N',
    help='Directory to save checkpoints in (once per epoch)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no-gpu',
    action='store_true',
    default=False,
    help='disables GPU usage even if a GPU is available')

FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
