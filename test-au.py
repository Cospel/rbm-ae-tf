from au import AutoEncoder
import tensorflow as tf
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
autoencoder = AutoEncoder(784, [100, 20],  [['rbmw1', 'rbmhb1'],
                                            ['rbmw2', 'rbmhb2']])

#autoencoder.restore_weights('./rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
#autoencoder.restore_weights('./rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)

for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(10)

  print(autoencoder.partial_fit(batch_xs))
  #print(autoencoder.transform(batch_xs))