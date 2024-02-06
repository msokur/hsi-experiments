import tensorflow as tf
import os
import tensorflow.keras as keras


TFR_SEED = 42


def get_dropout(net, dropout_value, name=None):
    seed = None
    if __is_deterministic():
        seed = TFR_SEED

    return keras.layers.Dropout(dropout_value, seed=seed, name=name)(net)


def get_initializers(kernel_mean=0.0, kernel_stddev=0.05, bias_value=0.1):
    kernel_initializer = "glorot_uniform"
    bias_initializer = "zeros"
    if __is_deterministic():
        kernel_initializer = keras.initializers.RandomNormal(mean=kernel_mean, stddev=kernel_stddev, seed=1337)
        bias_initializer = keras.initializers.Constant(value=bias_value)

    return kernel_initializer, bias_initializer


def set_tf_seed():
    if __is_deterministic():
        tf.random.set_seed(seed=TFR_SEED)


def __is_deterministic() -> bool:
    if os.environ.get("TF_DETERMINISTIC_OPS") is not None:
        if os.environ["TF_DETERMINISTIC_OPS"] == "1":
            return True

    return False
