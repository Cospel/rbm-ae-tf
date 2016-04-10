from rbm import RBM
import tensorflow as tf
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

# First RBM
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
rbmobject1 = RBM(784, 100, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.001)

# Train it
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(10)
  print(rbmobject1.partial_fit(batch_xs))

data_new = rbmobject1.transform(mnist.train.next_batch(10000)[0])
rbmobject1.save_weights('./rbmw1.chp')

# Second RBM
rbmobject2 = RBM(100, 20, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.001)

# Train it
for i in range(10000):
  # Transform features with first rbm to second rbm
  batch_xs, batch_ys = mnist.train.next_batch(10)
  batch_xs = rbmobject1.transform(batch_xs)
  print(rbmobject2.partial_fit(batch_xs))

rbmobject2.save_weights('./rbmw2.chp')




