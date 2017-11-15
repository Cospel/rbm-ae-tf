import os
from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_data
from utilsnn import show_image, min_max_scale
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_integer('epochs', 50, 'The number of training epochs')
flags.DEFINE_integer('batchsize', 30, 'The batch size')
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')

# ensure output dir exists
if not os.path.isdir('out'):
  os.mkdir('out')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX, teY = min_max_scale(trX, teX)

# RBMs
rbmobject1 = RBM(784, 900, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3)
rbmobject2 = RBM(900, 500, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3)
rbmobject3 = RBM(500, 250, ['rbmw3', 'rbvb3', 'rbmhb3'], 0.3)
rbmobject4 = RBM(250, 2,   ['rbmw4', 'rbvb4', 'rbmhb4'], 0.3)

if FLAGS.restore_rbm:
  rbmobject1.restore_weights('./out/rbmw1.chp')
  rbmobject2.restore_weights('./out/rbmw2.chp')
  rbmobject3.restore_weights('./out/rbmw3.chp')
  rbmobject4.restore_weights('./out/rbmw4.chp')

# Autoencoder
autoencoder = AutoEncoder(784, [900, 500, 250, 2], [['rbmw1', 'rbmhb1'],
                                                    ['rbmw2', 'rbmhb2'],
                                                    ['rbmw3', 'rbmhb3'],
                                                    ['rbmw4', 'rbmhb4']], tied_weights=False)

iterations = len(trX) / FLAGS.batchsize

# Train First RBM
print('first rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    rbmobject1.partial_fit(batch_xs)
  print(rbmobject1.compute_cost(trX))
  show_image("out/1rbm.jpg", rbmobject1.n_w, (28, 28), (30, 30))
rbmobject1.save_weights('./out/rbmw1.chp')

# Train Second RBM2
print('second rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    # Transform features with first rbm for second rbm
    batch_xs = rbmobject1.transform(batch_xs)
    rbmobject2.partial_fit(batch_xs)
  print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
  show_image("out/2rbm.jpg", rbmobject2.n_w, (30, 30), (25, 20))
rbmobject2.save_weights('./out/rbmw2.chp')

# Train Third RBM
print('third rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    # Transform features
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    rbmobject3.partial_fit(batch_xs)
  print(rbmobject3.compute_cost(rbmobject2.transform(rbmobject1.transform(trX))))
  show_image("out/3rbm.jpg", rbmobject3.n_w, (25, 20), (25, 10))
rbmobject3.save_weights('./out/rbmw3.chp')

# Train Third RBM
print('fourth rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    # Transform features
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    batch_xs = rbmobject3.transform(batch_xs)
    rbmobject4.partial_fit(batch_xs)
  print(rbmobject4.compute_cost(rbmobject3.transform(rbmobject2.transform(rbmobject1.transform(trX)))))
rbmobject4.save_weights('./out/rbmw4.chp')


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./out/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./out/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
autoencoder.load_rbm_weights('./out/rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)
autoencoder.load_rbm_weights('./out/rbmw4.chp', ['rbmw4', 'rbmhb4'], 3)

# Train Autoencoder
print('autoencoder')
for i in range(FLAGS.epochs):
  cost = 0.0
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    cost += autoencoder.partial_fit(batch_xs)
  print(cost)

autoencoder.save_weights('./out/au.chp')
autoencoder.load_weights('./out/au.chp')

fig, ax = plt.subplots()

print(autoencoder.transform(teX)[:, 0])
print(autoencoder.transform(teX)[:, 1])

plt.scatter(autoencoder.transform(teX)[:, 0], autoencoder.transform(teX)[:, 1], alpha=0.5)
plt.show()

raw_input("Press Enter to continue...")
plt.savefig('out/myfig')

