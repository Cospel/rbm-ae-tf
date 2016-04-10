import tensorflow as tf


class RBM(object):
    def __init__(self, n_input, n_hidden, layer_names, alpha=0.1, transfer_function=tf.nn.sigmoid):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.layer_names = layer_names

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

        # compute updates and add them to weights
        self.update_w = self.weights['w'].assign_add(alpha * (self.w_positive_grad - self.w_negative_grad))
        self.update_vb = self.weights['vb'].assign_add(alpha * tf.reduce_mean(self.x - self.v1, 0))
        self.update_hb = self.weights['hb'].assign_add(alpha * tf.reduce_mean(self.h0 - self.h1, 0))

        # sampling
        self.h_sample = self.sample_prob(transfer_function(tf.matmul(self.x, self.weights['w']) + self.weights['hb']))
        self.v_sample = self.sample_prob(
            transfer_function(tf.matmul(self.h_sample, tf.transpose(self.weights['w'])) + self.weights['vb']))

        # cost
        self.err_sum = tf.reduce_mean(tf.square(self.x - self.v_sample))

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w'] = tf.Variable(tf.zeros([self.n_input, self.n_hidden], dtype=tf.float32), name=self.layer_names[0])
        all_weights['vb'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name=self.layer_names[1])
        all_weights['hb'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name=self.layer_names[2])
        return all_weights

    def transform(self, X):
        return self.sess.run(self.h1, {self.x: X})

    def restore_weights(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def save_weights(self, path):
        saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
                                self.layer_names[1]: self.weights['vb'],
                                self.layer_names[2]: self.weights['hb']})
        save_path = saver.save(self.sess, path)

    def return_weights(self):
        return self.weights

    def return_hidden_weight_as_np(self, name):
        return self.weights[name].eval(self.sess)

    def partial_fit(self, batch_x):
        self.sess.run([self.update_w,
                       self.update_vb,
                       self.update_hb], {self.x: batch_x})

        return self.sess.run(self.err_sum, feed_dict={self.x: batch_x})
