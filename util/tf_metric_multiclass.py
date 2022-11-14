from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras import backend as K
import numpy as np


class F1_score(Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1 = self.add_weight('f1', initializer='zeros')
        self.precision = Precision()
        self.recall = Recall()
        self.reshape_array = reshape_array

    # @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = self.reshape_array(K.argmax(y_pred, axis=1), y_pred.shape)
        true = self.reshape_array(y_true, pred.shape)
        p = self.precision(true, pred)
        r = self.recall(true, pred)
        f1_s = 2 * ((p * r) / (p + r + K.epsilon()))

        self.f1.assign(f1_s)

    def result(self):
        return self.f1

    def reset_states(self):
        self.f1.assign(0)
        self.__reset_var()

    def __reset_var(self):
        self.precision.reset_states()
        self.recall.reset_states()


def reshape_array(arr, shape):
    new_arr = np.zeros(shape=(shape[0], 8))
    arr_indexes = np.reshape(arr, newshape=(shape[0], 1))
    np.put_along_axis(new_arr, arr_indexes, 1, axis=1)
    return new_arr
