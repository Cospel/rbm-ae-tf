from rbm import RBM
import tensorflow as tf
import numpy as np
import input_data
import pylab
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
rbmobject = RBM(784, 50, 0.001)

for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(10)
  print(rbmobject.partial_fit(batch_xs))
  #print(rbmobject.transform(batch_xs))

test = rbmobject.transform(mnist.train.next_batch(1000)[0])
#print (test)
#print (test[:,0])
#ax = pylab.subplot(111)
#ax.scatter(test[:,0],test[:,1], c='b')
#ax.figure.show()
#wait = input("PRESS ENTER TO CONTINUE.")


#print(rbmobject.return_hidden_weight('w'))
rbmobject.save_weights('./test.chp')


