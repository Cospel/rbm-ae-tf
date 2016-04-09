# refactored from https://www.snip2code.com/Snippet/1059693/RBM-procedure-using-tensorflow

import tensorflow as tf


class RBM(object):
    def __init__(self, n_input, n_hidden, alpha=0.1, transfer_function=tf.nn.sigmoid):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.h0 = self.sample_prob(transfer_function(tf.matmul(self.x, self.weights['w']) + self.weights['hb']))
        self.v1 = self.sample_prob(
            transfer_function(tf.matmul(self.h0, tf.transpose(self.weights['w'])) + self.weights['vb']))
        self.h1 = tf.nn.sigmoid(tf.matmul(self.v1, self.weights['w']) + self.weights['hb'])

        # compute gradients
        self.w_positive_grad = tf.matmul(tf.transpose(self.x), self.h0)
        self.w_negative_grad = tf.matmul(tf.transpose(self.v1), self.h1)

        # compute updates
        self.update_w = self.weights['w'] + alpha * (self.w_positive_grad - self.w_negative_grad)
        self.update_vb = self.weights['vb'] + alpha * tf.reduce_mean(self.x - self.v1, 0)
        self.update_hb = self.weights['hb'] + alpha * tf.reduce_mean(self.h0 - self.h1, 0)

        # sampling
        self.h_sample = self.sample_prob(transfer_function(tf.matmul(self.x, self.weights['w']) + self.weights['hb']))
        self.v_sample = self.sample_prob(
            transfer_function(tf.matmul(self.h_sample, tf.transpose(self.weights['w'])) + self.weights['vb']))

        # cost
        self.err = self.x - self.v_sample
        self.err_sum = tf.reduce_mean(self.err * self.err)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w'] = tf.Variable(tf.zeros([self.n_input, self.n_hidden], dtype=tf.float32), name='rbmw')
        all_weights['vb'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='rbmvb')
        all_weights['hb'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='rbmhb')
        return all_weights

    def transform(self, X):
        return self.sess.run(self.h1, {self.x: X})

    def restore_weights(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def save_weights(self, path):
        saver = tf.train.Saver({'rbmw': self.weights['w'],
                                'rbmvb': self.weights['vb'],
                                'rbmhb': self.weights['hb']})
        save_path = saver.save(self.sess, path)

    def return_weights(self):
        return self.weights

    def return_hidden_weight_as_np(self, name):
        return self.weights[name].eval(self.sess)

    def partial_fit(self, X):
        sess = self.sess
        n_w = sess.run(self.update_w, feed_dict={self.x: X})
        n_vb = sess.run(self.update_vb, feed_dict={self.x: X})
        n_hb = sess.run(self.update_hb, feed_dict={self.x: X})
        sess.run(self.weights['w'].assign(n_w))
        sess.run(self.weights['vb'].assign(n_vb))
        sess.run(self.weights['hb'].assign(n_hb))
        return sess.run(self.err_sum, feed_dict={self.x: X})
