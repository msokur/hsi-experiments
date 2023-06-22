from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
import tensorflow as tf


class F1_score(Metric):
    def __init__(self, num_classes, name="f1_score", average="weighted", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.f1 = None
        self.average_methode = None
        self.init_average(average)
        self.strategy = tf.distribute.get_strategy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_max = K.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred_max, dtype=tf.float32, num_classes=self.num_classes)
        tp = tf.linalg.diag_part(cm)
        fp = tf.math.reduce_sum(cm, 0) - tp
        fn = tf.math.reduce_sum(cm, 1) - tp
        tn = tf.math.reduce_sum(cm) - tp - fp - fn
        self.wrapp(self.average_methode, tp, fp, fn, tn)

    def macro(self, tp, fp, fn, tn):
        f1_s = tp / (tp + 0.5 * (fp + fn) + K.epsilon())
        return K.sum(f1_s) / self.num_classes

    @staticmethod
    def micro(tp, fp, fn, tn):
        tp_sum = K.sum(tp)
        fp_sum = K.sum(fp)
        fn_sum = K.sum(fn)
        f1_s = tp_sum / (tp_sum + 0.5 * (fp_sum + fn_sum) + K.epsilon())
        return f1_s

    @staticmethod
    def weighted(tp, fp, fn, tn):
        weights = (tp + fn) / (tp + fn + fp + tn)
        f1_s = tp / (tp + 0.5 * (fp + fn) + K.epsilon())
        f1_s *= weights
        return K.sum(f1_s)

    @staticmethod
    def multi(tp, fp, fn, tn):
        f1_s = tp / (tp + 0.5 * (fp + fn) + K.epsilon())
        return f1_s

    def wrapp(self, average_fn, tp, fp, fn, tn):
        f1_s = average_fn(tp, fp, fn, tn)
        if self.strategy is not None:
            f1_s = f1_s / self.strategy.num_replicas_in_sync
        self.f1.assign(f1_s)

    def init_average(self, average):
        self.average = average

        if average == "macro":
            self.average_methode = self.macro
        elif average == "micro":
            self.average_methode = self.micro
        elif average == "weighted":
            self.average_methode = self.weighted
        elif average == "multi":
            self.average_methode = self.multi
            self.f1 = self.add_weight("f1", shape=(self.num_classes,), initializer="zeros")
            return
        else:
            raise ValueError("Only the average Keywords: 'macro', 'micro', 'weighted' and 'multi' are allowed!")

        self.f1 = self.add_weight("f1", initializer="zeros")

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
        self.f1.assign(tf.zeros(shape=self.f1.shape))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_classes": self.num_classes, "average": self.average}
