from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

# First RBM
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

rbmobject1 = RBM(784, 100, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.001)

# Second RBM
rbmobject2 = RBM(100, 20, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.001)

# Autoencoder
autoencoder = AutoEncoder(784, [100, 20],  [['rbmw1', 'rbmhb1'],
                                            ['rbmw2', 'rbmhb2']],
                                           tied_weights=False)

# Train First RBM
epoch = 2
for i in range(epoch):
  for i in range(100):
    print(i)
    batch_xs, batch_ys = mnist.train.next_batch(10)
    cost = rbmobject1.partial_fit(batch_xs)
  print(rbmobject1.compute_cost(trX))
rbmobject1.save_weights('./rbmw1.chp')

# Train Second RBM
for i in range(100):
  print(i)
  # Transform features with first rbm for second rbm
  batch_xs, batch_ys = mnist.train.next_batch(10)
  batch_xs = rbmobject1.transform(batch_xs)
  cost = rbmobject2.partial_fit(batch_xs)
print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
rbmobject2.save_weights('./rbmw2.chp')

# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)

# Train Autoencoder
for i in range(500):
  batch_xs, batch_ys = mnist.train.next_batch(10)
  cost = autoencoder.partial_fit(batch_xs)

autoencoder.save_weights('./au.chp')
autoencoder.load_weights('./au.chp')
