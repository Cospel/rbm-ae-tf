from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_data
from utilsnn import show_image, min_max_scale
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX, teY = min_max_scale(trX, teX)

# RBMs
rbmobject1 = RBM(784, 900, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3)
rbmobject2 = RBM(900, 500, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3)
rbmobject3 = RBM(500, 250, ['rbmw3', 'rbvb3', 'rbmhb3'], 0.3)
rbmobject4 = RBM(250, 2,   ['rbmw4', 'rbvb4', 'rbmhb4'], 0.3)

#rbmobject1.restore_weights('./rbmw1.chp')
#rbmobject2.restore_weights('./rbmw2.chp')
#rbmobject3.restore_weights('./rbmw3.chp')
#rbmobject4.restore_weights('./rbmw4.chp')

# Autoencoder
autoencoder = AutoEncoder(784, [900, 500, 250, 2], [['rbmw1', 'rbmhb1'],
                                                    ['rbmw2', 'rbmhb2'],
                                                    ['rbmw3', 'rbmhb3'],
                                                    ['rbmw4', 'rbmhb4']], tied_weights=False)

epoch = 50
batchsize = 30
iterations = len(trX)/batchsize

# Train First RBM
print('first rbm')
for i in range(epoch):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    rbmobject1.partial_fit(batch_xs)
  print(rbmobject1.compute_cost(trX))
  show_image("1rbm.jpg", rbmobject1.n_w, (28, 28), (30, 30))
rbmobject1.save_weights('./rbmw1.chp')

# Train Second RBM2
print('second rbm')
for i in range(epoch):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    # Transform features with first rbm for second rbm
    batch_xs = rbmobject1.transform(batch_xs)
    rbmobject2.partial_fit(batch_xs)
  print(rbmobject2.compute_cost(rbmobject1.transform(trX)))
  show_image("2rbm.jpg", rbmobject2.n_w, (30, 30), (25, 20))
rbmobject2.save_weights('./rbmw2.chp')

# Train Third RBM
print('third rbm')
for i in range(epoch):
  for j in range(iterations):
    # Transform features
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    rbmobject3.partial_fit(batch_xs)
  print(rbmobject3.compute_cost(rbmobject2.transform(rbmobject1.transform(trX))))
  show_image("3rbm.jpg", rbmobject3.n_w, (25, 20), (25, 10))
rbmobject3.save_weights('./rbmw3.chp')

# Train Third RBM
print('fourth rbm')
for i in range(epoch):
  for j in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    # Transform features
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    batch_xs = rbmobject3.transform(batch_xs)
    rbmobject4.partial_fit(batch_xs)
  print(rbmobject4.compute_cost(rbmobject3.transform(rbmobject2.transform(rbmobject1.transform(trX)))))
rbmobject4.save_weights('./rbmw4.chp')


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
autoencoder.load_rbm_weights('./rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)
autoencoder.load_rbm_weights('./rbmw4.chp', ['rbmw4', 'rbmhb4'], 3)

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

fig, ax = plt.subplots()

print(autoencoder.transform(teX)[:,0])
print(autoencoder.transform(teX)[:,1])

plt.scatter(autoencoder.transform(teX)[:,0], autoencoder.transform(teX)[:,1], alpha=0.5)
plt.show()

raw_input("Press Enter to continue...")
plt.savefig('myfig')

