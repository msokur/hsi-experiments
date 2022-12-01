import tensorflow as tf
from tensorflow import keras

from configuration.get_config import WITHOUT_RANDOMNESS


def get_dropout(net, dropout_value):
    if WITHOUT_RANDOMNESS:
        net = tf.keras.layers.Dropout(dropout_value, seed=3)(net)
    else:
        net = tf.keras.layers.Dropout(dropout_value)(net)
    return net


def get_inizializers():
    kernel_initializer = 'glorot_uniform'
    bias_initializer = 'zeros'
    if WITHOUT_RANDOMNESS:
        kernel_initializer = keras.initializers.RandomNormal(seed=1337)
        bias_initializer = keras.initializers.Constant(value=0.1)

    return kernel_initializer, bias_initializer
