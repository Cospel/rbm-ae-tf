from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_data
from utilsnn import show_image, min_max_scale
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX, teY = min_max_scale(trX, teX)

rbmobject1 = RBM(784, 500, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.1)
rbmobject2 = RBM(500, 200, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.1)
rbmobject3 = RBM(200, 2, ['rbmw3', 'rbvb3', 'rbmhb3'], 0.1)

# Autoencoder
autoencoder = AutoEncoder(784, [500, 200, 2], [['rbmw1', 'rbmhb1'],
                                              ['rbmw2', 'rbmhb2'],
                                              ['rbmw3', 'rbmhb3']],tied_weights=True)

epoch = 30
batchsize = 100
iterations = len(trX)/batchsize


# Train First RBM
print('first rbm')
for i in range(epoch):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    rbmobject1.partial_fit(batch_xs)
  print(rbmobject1.compute_cost(trX))
  show_image("1rbm.jpg", rbmobject1.n_w, (28, 28), (10, 10))
rbmobject1.save_weights('./rbmw1.chp')

# Train Second RBM
print('second rbm')
for i in range(epoch):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    # Transform features with first rbm for second rbm
    batch_xs = rbmobject1.transform(batch_xs)
    rbmobject2.partial_fit(batch_xs)
  print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
  show_image("2rbm.jpg", rbmobject2.n_w, (25, 20), (20, 20))
rbmobject2.save_weights('./rbmw2.chp')

# Train Third RBM
print('third rbm')
for i in range(epoch):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    # Transform features
    batch_xs, batch_ys = mnist.train.next_batch(10)
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    rbmobject3.partial_fit(batch_xs)
  print(rbmobject3.compute_cost(rbmobject2.transform(rbmobject1.transform(trX))))
rbmobject3.save_weights('./rbmw3.chp')

# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
autoencoder.load_rbm_weights('./rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)

# Train Autoencoder
print('autoencoder')
for i in range(epoch):
  cost = 0.0
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    cost += autoencoder.partial_fit(batch_xs)
  print(cost)

autoencoder.save_weights('./au.chp')
autoencoder.load_weights('./au.chp')
