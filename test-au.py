from au import AutoEncoder
import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
autoencoder = AutoEncoder(784, [50], 0.01)
#autoencoder.restore_weights('./test.chp')
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(10)

  print(autoencoder.partial_fit(batch_xs))
