# rbm-ae-tf
Tensorflow implementation of Restricted Boltzman Machine and Autoencoder for layerwise pretraining of Deep Autoencoders with RBM. Idea is to first create RBMs for pretraining weights for autoencoder. Then weigts for autoencoder are loaded and autoencoder is trained again. In this implementation you can also use tied weights for autoencoder.

More about pretraining of weights in this paper:

##### [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/science.pdf)

```python
from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

# First RBM
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
rbmobject1 = RBM(784, 100, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.001)

# Second RBM
rbmobject2 = RBM(100, 20, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.001)

# Autoencoder
autoencoder = AutoEncoder(784, [100, 20],  [['rbmw1', 'rbmhb1'],
                                            ['rbmw2', 'rbmhb2']],
                                            tied_weights=True)

# Train First RBM
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(10)
  cost = rbmobject1.partial_fit(batch_xs)

data_new = rbmobject1.transform(mnist.train.next_batch(10000)[0])
rbmobject1.save_weights('./rbmw1.chp')

# Train Second RBM
for i in range(10000):
  # Transform features with first rbm to second rbm
  batch_xs, batch_ys = mnist.train.next_batch(10)
  batch_xs = rbmobject1.transform(batch_xs)
  cost = rbmobject2.partial_fit(batch_xs)

# Load RBM weights to Autoencoder
autoencoder.restore_weights('./rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.restore_weights('./rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)

# Train Autoencoder
for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(10)
  cost = autoencoder.partial_fit(batch_xs)
```

I was inspired with these implementations but I need to refactor them and improve them(BIG THANKS to THEM). I tried to use also similar api as it is in [tensorflow/models](https://github.com/tensorflow/models):
> [LINK1](https://www.snip2code.com/Snippet/1059693/RBM-procedure-using-tensorflow)
> [LINK2](https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5)


Feel free to make updates, repairs. You can enhance implementation with some tips from:
> [Practical Guide to training RBM](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
