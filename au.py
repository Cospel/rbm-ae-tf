import tensorflow as tf
from utilsnn import xavier_init


class AutoEncoder(object):
    def __init__(self, input_size, layer_sizes, layer_names, optimizer=tf.train.AdamOptimizer(),
                 transfer_function=tf.nn.sigmoid):
        # Build the encoding layers
        self.x = tf.placeholder("float", [None, input_size])
        next_layer_input = self.x

        assert len(layer_sizes) == len(layer_names)

        self.encoding_matrices = []
        self.encoding_biases = []
        for i in range(len(layer_sizes)):
            dim = layer_sizes[i]
            input_dim = int(next_layer_input.get_shape()[1])

            # Initialize W using xavier initialization]
            W = tf.Variable(xavier_init(input_dim, dim, transfer_function), name=layer_names[i][0])

            # Initialize b to zero
            b = tf.Variable(tf.zeros([dim]), name=layer_names[i][1])

            # We are going to use tied-weights so store the W matrix for later reference.
            self.encoding_matrices.append(W)
            self.encoding_biases.append(b)

            output = transfer_function(tf.matmul(next_layer_input, W) + b)

            # the input into the next layer is the output of this layer
            next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        self.encoded_x = next_layer_input

        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        self.encoding_matrices.reverse()

        for i, dim in enumerate(layer_sizes[1:] + [int(self.x.get_shape()[1])]):
            # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
            W = tf.transpose(self.encoding_matrices[i])
            b = tf.Variable(tf.zeros([dim]))
            output = transfer_function(tf.matmul(next_layer_input, W) + b)
            next_layer_input = output

        # i need to reverse the encoding matrices back for loading weights
        self.encoding_matrices.reverse()

        # the fully encoded and reconstructed value of x is here:
        self.reconstructed_x = next_layer_input

        # compute cost
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        self.optimizer = optimizer.minimize(self.cost)

        # initalize variables
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def transform(self, X):
        return self.sess.run(self.encoded_x, {self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.reconstructed_x, feed_dict={self.x: X})

    def restore_weights(self, path, layer_names, layer):
        saver = tf.train.Saver({layer_names[0]: self.encoding_matrices[layer]},
                               {layer_names[1]: self.encoding_biases[layer]})
        saver.restore(self.sess, path)

    def save_weights(self, path):
        pass

    def return_weights(self):
        pass

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost
