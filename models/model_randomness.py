import tensorflow as tf
import os
import tensorflow.keras as keras


def get_dropout(net, dropout_value, name=None):
    if os.environ["TF_DETERMINISTIC_OPS"] == "1":
        net = keras.layers.Dropout(dropout_value, seed=3, name=name)(net)
    else:
        net = keras.layers.Dropout(dropout_value, name=name)(net)
    return net


def get_inizializers():
    kernel_initializer = "glorot_uniform"
    bias_initializer = "zeros"
    if os.environ["TF_DETERMINISTIC_OPS"] == "1":
        kernel_initializer = keras.initializers.RandomNormal(seed=1337)
        bias_initializer = keras.initializers.Constant(value=0.1)

    return kernel_initializer, bias_initializer


def set_tf_seed():
    if os.environ["TF_DETERMINISTIC_OPS"] == "1":
        tf.random.set_seed(seed=1)
