from array import array
import tensorflow as tf
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras import backend as K


class F1_score_multiclass(Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1 = self.add_weight('f1', shape=(num_classes,), initializer='zeros')
        self.precision = Precision()
        self.recall = Recall()

    # @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        f1_s = array('f', [])
        pred = K.argmax(y_pred, axis=-1)

        for class_ in range(self.num_classes):
            pred_bool = K.equal(pred, class_)
            true_bool = K.equal(y_true, class_)

            pred_float = K.cast(pred_bool, 'float32')
            true_float = K.cast(true_bool, 'float32')

            p = self.precision(true_float, pred_float)
            r = self.recall(true_float, pred_float)

            f1_class = 2 * ((p * r) / (p + r + K.epsilon()))
            f1_s.append(f1_class)
            self.__reset_var()

        self.f1.assign(f1_s)

    def result(self):
        return self.f1

    def reset_states(self):
        self.f1.assign(array('f', [0. for i in range(self.num_classes)]))
        self.__reset_var()

    def __reset_var(self):
        self.precision.reset_states()
        self.recall.reset_states()