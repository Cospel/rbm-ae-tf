import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from PIL import Image
from util import tile_raster_images


def show_image(path, n_w, img_shape, tile_shape):
    image = Image.fromarray(
        tile_raster_images(X=n_w.T, img_shape=img_shape, tile_shape=tile_shape, tile_spacing=(1, 1))
    )
    image.save(path)


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def min_max_scale(X_train, X_test):
    preprocessor = prep.MinMaxScaler().fit(np.concatenate((X_train, X_test), axis=0))
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def mean_normalization(X_train, X_test):
    data = np.concatenate((X_train, X_test), axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def xavier_init(fan_in, fan_out, function):
    if function is tf.nn.sigmoid:
        low = -4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    elif function is tf.nn.tanh:
        low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
