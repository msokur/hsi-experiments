from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
import tensorflow as tf


class F1_score(Metric):
    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.f1 = self.add_weight("f1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_threshold = K.cast(K.greater_equal(y_pred, self.threshold), dtype=tf.float32)

        tp = K.sum(y_true * y_pred_threshold)
        fp = K.sum(K.clip(y_pred_threshold - y_true, 0, 1))
        fn = K.sum(K.clip(y_true - y_pred_threshold, 0, 1))
        self.f1.assign(tp / (tp + 0.5 * (fp + fn) + K.epsilon()))

    def merge_state(self, metrics):
        length = len(metrics) + 1
        result = self.f1.numpy()
        for metric in metrics:
            if len(self.weights) == metric.weights.__len__():
                result += metric.f1.numpy()
            else:
                raise ValueError("Can not merge state!")
        self.f1.assign(result / length)

    def result(self):
        return self.f1

    def reset_state(self):
        self.f1.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
